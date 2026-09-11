"""T1c — 결재선 **예측** (leave-one-out).

T1b가 "문서에서 결재선을 읽어내기"였다면 T1c는 **"아직 없는 결재선을 제안하기"** 다.
회사 기능 ②(결재라인 자동 삽입)의 리허설이고, 고객 메일의
"상단은 조직정보, 중간·하단은 품의 내용" 구조를 그대로 흉내 낸다.

방법
    같은 부서 문서가 셋씩 있는 묶음에서 한 건을 빼고(leave-one-out)
    나머지 2건의 (제목·본문·결재선) + 대상 문서의 (제목·본문) 으로 대상 결재선을 예측한다.
    정답은 T1b 와 같은 ``approval_gold.tsv``.

    묶음: 법무부 총무과 3건, 경기도 경기도서관 3건 → 6케이스.

    **주의** 두 묶음의 성격이 다르다. 경기도서관 3건은 실제로 같은 조직이라
    결재선이 거의 같지만, 법무부 "총무과" 3건은 부서명만 같고 실제 결재 조직이
    서로 다르다(출입국·외국인청 / 교정기관 / 안양교도소). 그래서 묶음별 점수를
    따로 낸다. 6건을 뭉쳐 평균 내면 baseline 숫자가 뭉개진다.

개인정보
    프롬프트에는 제목·본문·실명이 들어가지만 **저장하지 않는다.**
    결과 JSON에는 케이스별 점수와 집계만 남긴다. 프롬프트 템플릿은
    ``t1c_prompt.md`` 에 값 없이 따로 둔다.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
import typer

from who_approved_this.collect.opengokr_local import OpenGoKrLocalCollector
from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.evaluate.approval_match import (
    ApprovalMatchEvaluator,
    load_gold,
    load_orgs,
)
from who_approved_this.evaluate.rules_diff import score_predictions_by_source
from who_approved_this.tracks.t1c_predictors import (
    BaselineCopy,
    BaselineVote,
    LocalLLM,
    Predictor,
    VoteTitlesLLMNames,
)
from who_approved_this.tracks.t1d_draft import mask_names

TRACK = "t1c"

#: leave-one-out 묶음 최소 크기. 참고 1건 + 대상 1건.
MIN_GROUP = 2

#: 참고 문서를 몇 건까지 주는지 바꿔가며 재는 값("과거 몇 건이면 충분한가").
REF_SWEEP = (1, 3, 7)

#: 재표집 검증에서 한 대상에 대해 참고 문서 조합을 몇 번 바꿔 볼지.
RESAMPLE_ROUNDS = 5

#: 재표집에서 쓸 참고 문서 수(전체에서 이만큼 무작위로 고른다).
RESAMPLE_REFS = 5


def load_docs(data_dir: Path) -> list[dict[str, Any]]:
    """PDF 본문(텍스트 레이어)과 gold 결재선을 문서별로 모은다."""
    gold = load_gold(data_dir / "approval_gold.tsv")
    orgs = load_orgs(data_dir / "approval_gold.tsv")
    docs: list[dict[str, Any]] = []
    for pdf_path, row in OpenGoKrLocalCollector(data_dir).items():
        key = row["저장파일명"]
        if key not in gold:
            continue
        with pymupdf.open(pdf_path) as doc:
            body = "\n".join(doc[i].get_text(sort=True) for i in range(doc.page_count))
        cells = [
            {"role": role, "title": c["title"], "name": c["name"]}
            for role, items in gold[key].items()
            for c in items
        ]
        docs.append(
            {**row, "org": orgs.get(key, ""), "body": body, "cells": cells, "pdf": pdf_path}
        )
    return docs


def groups_of(docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """**실제 조직**으로 묶는다(담당부서명이 아니라).

    포털의 ``담당부서`` 로 묶으면 법무부 "총무과" 8건이 한 묶음이 되는데, 실제로는
    안양교도소·대구합동청사·서울출입국청 등 **6개 조직**이 섞여 있다. 직위 체계부터
    달라 예측이 성립하지 않는다. 조직 키는 문서 하단 주소를 보고 gold 에 적어 둔 값이다.

    조직당 1건뿐이면 참고할 과거 문서가 없으므로 예측 대상에서 뺀다.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        out.setdefault(d["org"] or d["담당부서"], []).append(d)
    return {k: v for k, v in out.items() if len(v) >= MIN_GROUP}


def all_names(docs: list[dict[str, Any]], data_dir: Path) -> set[str]:
    """가릴 이름 — gold 결재선의 이름 + 명부. 이름을 본문에서 추정하지 않는다."""
    names = {c["name"] for d in docs for c in d["cells"] if c["name"] not in ("", "-", "?")}
    roster = data_dir / "roster.tsv"
    if roster.is_file():
        with roster.open(encoding="utf-8") as fh:
            names |= {r["name"] for r in csv.DictReader(fh, delimiter="\t") if r.get("name")}
    return {n for n in names if len(n) >= 2}


def cell_outcomes(pred: list[dict[str, Any]], gold_doc: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    """칸별 정오와 근거 태그. 값(직위·이름)은 담지 않는다."""
    from who_approved_this.evaluate.approval_match import normalize_token

    out = []
    for role in ("기안", "검토", "결재", "협조"):
        p = [c for c in pred if c["role"] == role]
        for i, c in enumerate(p):
            g = gold_doc[role][i] if i < len(gold_doc[role]) else None
            t_src, _, n_src = c.get("evidence", "?/?").partition("/")
            name_ok = None
            if g is not None and g["name"] != "?":
                name_ok = normalize_token(c["name"]) == normalize_token(g["name"])
            out.append({"role": role, "title_src": t_src, "name_src": n_src or "?",
                        "title_ok": g is not None and normalize_token(c["title"]) == normalize_token(g["title"]),
                        "name_ok": name_ok if g is not None else False})
    return out


def run(
    model: str | None = None,
    max_docs: int | None = None,
    resample: int = 0,
    rules: tuple[str, ...] = ("induced", "manual"),
    mask_target: bool = True,
    extra_predictors: list[Predictor] | None = None,
    baselines: bool = True,
) -> dict[str, Any]:
    """T1c를 돌리고 결과 JSON을 남긴다. ``model`` 을 주면 llm_local 도 함께 돈다.

    ``mask_target`` — 대상 문서 본문의 사람 이름을 가린다. 본문 텍스트 레이어에 결재란이
    들어 있어 12건 전부 **정답 이름이 프롬프트에 보였다**(2026-09-11 확인). 결재 전 문서에는
    결재란이 없으므로 가리는 쪽이 실제 조건이다.
    """
    data_dir = track_data_dir("t1")
    results_dir = track_results_dir(TRACK)

    gold_path = data_dir / "approval_gold.tsv"
    if not gold_path.is_file():
        raise FileNotFoundError(f"gold 가 없다: {gold_path}")

    docs = load_docs(data_dir)
    groups = groups_of(docs)
    evaluator = ApprovalMatchEvaluator(load_gold(gold_path))

    predictors: list[Predictor] = [BaselineCopy(), BaselineVote()] if baselines else []
    if model and baselines:
        predictors.append(LocalLLM(model))
        predictors.append(VoteTitlesLLMNames(model))
    # 늦은 import — t1e_worktype 가 이 모듈의 load_docs 를 부르고, layered 가 t1e 를 부르는 순환을 끊는다
    from who_approved_this.tracks.t1c_layered import LayeredPredictor

    for r in rules:
        predictors.append(LayeredPredictor(model, rules=r))
    predictors.extend(extra_predictors or [])
    names = all_names(docs, data_dir) if mask_target else set()

    started = time.perf_counter()
    results: dict[str, Any] = {}
    for predictor in predictors:
        cases: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for dept, members in sorted(groups.items()):
            for target in members:
                refs = [d for d in members if d["doc_id"] != target["doc_id"]]
                seen = {**target, "body": mask_names(target["body"], names)} if names else target
                c0 = time.perf_counter()
                predicted = predictor.predict(refs, seen)
                case_sec = round(time.perf_counter() - c0, 2)
                scores = evaluator.evaluate({"cells": predicted}, target)
                gold_doc = evaluator.gold[target["저장파일명"]]
                g_first = next((c["title"] for c in gold_doc["검토"]), None)
                p_first = next((c["title"] for c in predicted if c["role"] == "검토"), None)
                cases.append(
                    {
                        "group": dept,
                        "doc_id": target["doc_id"],
                        "elapsed_sec": case_sec,
                        "refs": len(refs),
                        # 값은 담지 않는다 — 개수와 일치율만.
                        "gold_cells": int(sum(scores[f"role_{r}_gold_cells"] for r in
                                              ("기안", "검토", "결재", "협조"))),
                        "pred_cells": len(predicted),
                        "title_accuracy": round(scores["title_accuracy"], 4),
                        "name_accuracy": round(scores["name_accuracy"], 4),
                        "cell_count_match": round(scores["cell_count_match"], 4),
                        "coop_gold": int(scores["role_협조_gold_cells"]),
                        "coop_pred": int(scores["role_협조_pred_cells"]),
                        "coop_hits": int(scores["role_협조_title_hits"]),
                        "coop_exact": bool(scores["role_협조_count_match"]
                                           and scores["role_협조_title_hits"] == scores["role_협조_gold_cells"]),
                        "first_review_ok": g_first == p_first if g_first else None,
                        # 직위 순서만(이름 없음) — 예측기끼리 케이스별로 비교할 때 쓴다
                        "pred_titles": [f"{c['role']}:{c['title']}" for c in predicted],
                        "cells": cell_outcomes(predicted, gold_doc) if any(
                            "evidence" in c and "/" in c.get("evidence", "") for c in predicted) else [],
                        "extra": getattr(predictor, "last_trace", None),
                        "failures": [
                            {"role": r, "reason": why} for r, why in evaluator.failures
                        ],
                    }
                )
        results[predictor.name] = {
            "json_retries": getattr(predictor, "retries", 0),
            "json_failures": getattr(predictor, "failures", 0),
            "llm_calls": getattr(predictor, "llm_calls", None),
            "by_evidence": score_predictions_by_source([x for c in cases for x in c["cells"]]),
            "cases": cases,
            "summary": _summarize(cases),
            "by_group": {
                g: _summarize([c for c in cases if c["group"] == g])
                for g in sorted(groups)
            },
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    sweep = _ref_sweep(groups, evaluator)
    robustness = (
        _robustness(groups, evaluator, model, resample)
        if resample and model
        else {"skipped": "재표집 미실행"}
    )

    now = datetime.now()
    report = {
        "track": TRACK,
        "run_at": now.isoformat(timespec="seconds"),
        "method": "leave-one-out (같은 담당부서 3건 중 2건을 참고로 1건 예측)",
        "ground_truth": "approval_gold.tsv (사람이 직접 만든 정답, 리포 밖)",
        "note": "프롬프트에 들어간 제목·본문·실명은 저장하지 않는다. 점수와 개수만.",
        "target_names_masked": bool(names),
        "prompt_template": "src/who_approved_this/tracks/t1c_prompt.md",
        "model": model,
        "groups": {g: len(v) for g, v in sorted(groups.items())},
        "predictors": results,
        "ref_sweep": sweep,
        "robustness": robustness,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    out_path = results_dir / f"{now:%Y%m%d-%H%M}.json"
    if out_path.exists():
        out_path = results_dir / f"{now:%Y%m%d-%H%M%S}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["results_path"] = str(out_path)
    return report


def _ref_sweep(
    groups: dict[str, list[dict[str, Any]]], evaluator: ApprovalMatchEvaluator
) -> dict[str, Any]:
    """참고 문서 수를 바꿔가며 baseline_vote 점수 변화를 본다.

    "과거 문서 몇 건이면 충분한가"에 대한 답. 참고가 1건이면 vote 는 복사와 같아진다.

    **같은 케이스 집합으로만 비교한다.** 참고를 최대 개수까지 줄 수 있는 묶음만 쓴다.
    참고 1건일 때만 작은 묶음이 끼면 케이스가 달라져 추세가 오염된다.
    """
    eligible = {
        k: v for k, v in groups.items() if len(v) - 1 >= max(REF_SWEEP)
    }
    out: dict[str, Any] = {"groups": sorted(eligible)}
    for n_refs in REF_SWEEP:
        cases: list[dict[str, Any]] = []
        for dept, members in sorted(eligible.items()):
            for target in members:
                refs = _recent_first_docs(
                    [d for d in members if d["doc_id"] != target["doc_id"]]
                )[:n_refs]
                scores = evaluator.evaluate(
                    {"cells": BaselineVote().predict(refs, target)}, target
                )
                cases.append(
                    {
                        "group": dept,
                        "title_accuracy": round(scores["title_accuracy"], 4),
                        "name_accuracy": round(scores["name_accuracy"], 4),
                        "cell_count_match": round(scores["cell_count_match"], 4),
                    }
                )
        out[str(n_refs)] = _summarize(cases)
    return out


def _recent_first_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """생산일자 내림차순. 참고 문서를 최근 것부터 자르기 위해."""
    return sorted(
        docs, key=lambda d: (d.get("생산일자", ""), d.get("문서번호", "")), reverse=True
    )


def _robustness(
    groups: dict[str, list[dict[str, Any]]],
    evaluator: ApprovalMatchEvaluator,
    model: str,
    rounds: int = RESAMPLE_ROUNDS,
) -> dict[str, Any]:
    """참고 문서 **조합을 바꿔가며** 세 예측기를 다시 재는 재표집 검증.

    새 문서를 더 모은 게 아니라 **같은 데이터에서 측정 지점을 늘린 것**이다.
    독립 표본이 아니므로 신뢰구간으로 쓸 수 없고, "이 우열이 참고 문서 조합에
    따라 뒤집히는가"만 본다. 뒤집히면 12케이스 결과는 우연이었다는 뜻이다.

    참고를 넉넉히 줄 수 있는 묶음(문서 6건 이상)만 대상으로 한다.
    """
    import random

    eligible = {k: v for k, v in groups.items() if len(v) - 1 >= RESAMPLE_REFS}
    if not eligible:
        return {"skipped": f"참고 {RESAMPLE_REFS}건을 줄 수 있는 묶음이 없다"}

    rng = random.Random(20260909)
    rows: dict[str, list[dict[str, float]]] = {}
    for round_no in range(rounds):
        for name, make in (
            ("baseline_vote", lambda: BaselineVote()),
            ("llm_local", lambda: LocalLLM(model)),
            ("vote_titles+llm_names", lambda: VoteTitlesLLMNames(model)),
        ):
            rows.setdefault(name, [])
        for group, members in sorted(eligible.items()):
            for target in members:
                pool = [d for d in members if d["doc_id"] != target["doc_id"]]
                refs = rng.sample(pool, RESAMPLE_REFS)
                for name, predictor in (
                    ("baseline_vote", BaselineVote()),
                    ("llm_local", LocalLLM(model)),
                    ("vote_titles+llm_names", VoteTitlesLLMNames(model)),
                ):
                    sc = evaluator.evaluate({"cells": predictor.predict(refs, target)}, target)
                    rows[name].append(
                        {
                            "round": float(round_no),
                            "title": sc["title_accuracy"],
                            "name": sc["name_accuracy"],
                        }
                    )

    out: dict[str, Any] = {
        "groups": sorted(eligible),
        "rounds": rounds,
        "refs_per_round": RESAMPLE_REFS,
        "measurements_per_predictor": len(next(iter(rows.values()))),
        "by_predictor": {},
    }
    for name, vals in rows.items():
        n = len(vals) or 1
        out["by_predictor"][name] = {
            "title_mean": round(sum(v["title"] for v in vals) / n, 4),
            "name_mean": round(sum(v["name"] for v in vals) / n, 4),
        }

    # 라운드별로 결합이 두 단독을 이겼는지 세어 본다(우열이 뒤집히는지).
    per_round: list[dict[str, Any]] = []
    for r in range(rounds):
        pick = lambda k: [v for v in rows[k] if v["round"] == r]
        combo, vote, llm = pick("vote_titles+llm_names"), pick("baseline_vote"), pick("llm_local")
        m = lambda vs, f: sum(v[f] for v in vs) / max(len(vs), 1)
        per_round.append(
            {
                "round": r,
                "combo_name": round(m(combo, "name"), 4),
                "vote_name": round(m(vote, "name"), 4),
                "llm_name": round(m(llm, "name"), 4),
                "combo_title": round(m(combo, "title"), 4),
                "vote_title": round(m(vote, "title"), 4),
                "combo_beats_both_on_name": bool(
                    m(combo, "name") >= m(vote, "name") and m(combo, "name") >= m(llm, "name")
                ),
            }
        )
    out["per_round"] = per_round
    out["rounds_combo_best_on_name"] = sum(
        1 for r in per_round if r["combo_beats_both_on_name"]
    )
    return out


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """케이스 묶음의 평균. ``title_accuracy`` 가 곧 "직위만" 점수다."""
    n = len(cases) or 1
    return {
        "cases": len(cases),
        "title_accuracy_mean": round(sum(c["title_accuracy"] for c in cases) / n, 4),
        "name_accuracy_mean": round(sum(c["name_accuracy"] for c in cases) / n, 4),
        "cell_count_match_mean": round(sum(c["cell_count_match"] for c in cases) / n, 4),
        "cases_title_perfect": sum(1 for c in cases if c["title_accuracy"] == 1.0),
        # 협조: gold 협조 칸 중 맞힌 칸 / 협조가 칸 수·직위 모두 맞은 케이스
        "coop_gold_cells": sum(c.get("coop_gold", 0) for c in cases),
        "coop_cell_hits": sum(c.get("coop_hits", 0) for c in cases),
        "coop_extra_cells": sum(max(c.get("coop_pred", 0) - c.get("coop_gold", 0), 0) for c in cases),
        "coop_case_exact": sum(1 for c in cases if c.get("coop_exact")),
        "first_review_ok": sum(1 for c in cases if c.get("first_review_ok")),
        "first_review_scored": sum(1 for c in cases if c.get("first_review_ok") is not None),
        "sec_per_case_mean": round(
            sum(c.get("elapsed_sec", 0.0) for c in cases) / n, 2
        ),
    }


def report(rep: dict[str, Any]) -> None:
    """콘솔 출력. 실패는 케이스 id / 역할 / 사유로만 — 값은 찍지 않는다."""
    typer.echo(f"[t1c] {rep['method']}")
    typer.echo(f"      묶음: {rep['groups']}  모델: {rep['model'] or '(llm 미실행)'}")
    typer.echo("")
    typer.echo(f"      대상 본문 이름 가림: {rep.get('target_names_masked')}")
    typer.echo(f"  {'예측기':30} {'직위만':>7} {'이름':>7} {'칸수':>7} {'협조칸':>7} {'협조케이스':>6} {'첫검토':>6}  직위전부맞은케이스")
    for name, data in rep["predictors"].items():
        s = data["summary"]
        typer.echo(
            f"  {name:30} {s['title_accuracy_mean']:6.1%} {s['name_accuracy_mean']:6.1%}"
            f" {s['cell_count_match_mean']:6.1%}  {s['coop_cell_hits']}/{s['coop_gold_cells']}(+{s['coop_extra_cells']})"
            f"  {s['coop_case_exact']:2}/{s['cases']}  {s['first_review_ok']}/{s['first_review_scored']}"
            f"   {s['cases_title_perfect']}/{s['cases']}"
            f"   케이스당 {s['sec_per_case_mean']:5.1f}s"
            + (
                f"  (JSON 재시도 {data['json_retries']}, 실패 {data['json_failures']})"
                if data.get("json_retries") or data.get("json_failures")
                else ""
            )
        )
    typer.echo("")
    typer.echo("  묶음별 (성격이 달라 따로 본다)")
    for name, data in rep["predictors"].items():
        for group, s in data["by_group"].items():
            typer.echo(
                f"    {name:20} {group:10} 직위 {s['title_accuracy_mean']:6.1%}"
                f" | 이름 {s['name_accuracy_mean']:6.1%} | 칸수 {s['cell_count_match_mean']:6.1%}"
                f" ({s['cases_title_perfect']}/{s['cases']})"
            )
    sweep = rep.get("ref_sweep", {})
    if sweep:
        typer.echo("")
        typer.echo(
            f"  참고 문서 수별 baseline_vote — {', '.join(sweep.get('groups', []))} 만"
            " (같은 케이스 집합)"
        )
        for n, s in sweep.items():
            if n == "groups" or not s["cases"]:
                continue
            typer.echo(
                f"    참고 {n:>2}건: 직위 {s['title_accuracy_mean']:6.1%}"
                f" | 이름 {s['name_accuracy_mean']:6.1%}"
                f" | 칸수 {s['cell_count_match_mean']:6.1%}  ({s['cases']}케이스)"
            )

    rb = rep.get("robustness", {})
    if "by_predictor" in rb:
        typer.echo("")
        typer.echo(
            f"  재표집 검증 — {', '.join(rb['groups'])}, 참고 {rb['refs_per_round']}건을"
            f" {rb['rounds']}라운드 무작위 재추출 (예측기당 {rb['measurements_per_predictor']}회 측정)"
        )
        for name, m in rb["by_predictor"].items():
            typer.echo(f"    {name:24} 직위 {m['title_mean']:6.1%} | 이름 {m['name_mean']:6.1%}")
        typer.echo(
            f"    → 결합이 이름에서 두 단독을 모두 이긴 라운드: "
            f"{rb['rounds_combo_best_on_name']}/{rb['rounds']}"
        )

    typer.echo("")
    typer.echo(f"  → {rep['results_path']}  ({rep['elapsed_sec']}s)")
