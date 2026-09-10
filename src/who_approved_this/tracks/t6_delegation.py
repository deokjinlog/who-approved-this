"""T6 — 가평군 공식 전결 규정으로 결재선 자동 지정 검증.

T1f(경기도서관)는 전결표가 gold 에서 귀납한 것이었다. 여기는 **공식 별표**가 있다.
기안자 직위와 소속 팀은 입력(로그인 사용자), 나머지는 규정·조직도로 만든다.

세 가지를 따로 잰다. 섞으면 어디서 틀렸는지 안 보인다.

    경로(대리 표 없음)   조직도 슬롯 체인만. 종점은 gold 로 준다
    경로(대리 표 있음)   + 직무대리 표(인사 데이터). 종점은 gold 로 준다
    에이전트            + 별표 조회로 종점까지 스스로 정한다

**주의: 직무대리 표는 gold 에서 귀납했다.** "대리 표 있음" 점수는 일반화가 아니라
표현력 확인이다 — 슬롯 + 대리 세 줄로 30건이 설명되는지. 회사에서는 인사 시스템이 준다.

gold 30건의 최종 결재가 전부 군수(권한 기준)라서, 종점 정확도는 예측기를 가르지 못한다.
원문정보공개가 높은 결재 문서 위주로 올라오는 선택 편향이다. 그래서 종점은
"별표 최소 레벨 vs 실제"를 같다 / 상향 결재 / 위반 셋으로 센다.
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
import typer
from rank_bm25 import BM25Okapi

from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.parse.official_line import LEVEL_NAMES, RULES, OfficialLineBuilder
from who_approved_this.parse.slot_chain import build_slot_chain, title_level
from who_approved_this.tracks.t1d_retrieval import tokenize

TRACK = "t6"

def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t)


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            d = docs.setdefault(
                r["no"], {"문서번호": r["문서번호"], "org": r["org"], "date": r["date"], "cells": []}
            )
            d["cells"].append({"role": r["role"], "title": r["title"], "note": r["note"]})
    return docs


def score(pred: list[str], gold: list[str]) -> dict[str, Any]:
    n = max(len(pred), len(gold)) or 1
    hits = sum(_norm(p) == _norm(g) for p, g in zip(pred, gold))
    return {
        "title_accuracy": round(hits / n, 4),
        "exact": len(pred) == len(gold) and hits == len(gold),
        "cell_count_match": len(pred) == len(gold),
    }


def load_row_gold(path: Path) -> dict[str, set[int]]:
    """``row_gold.tsv`` — 문서번호 → 맞는 별표 행(rid) 집합. 빈 집합은 '맞는 행 없음'."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {
            r["문서번호"]: {int(x) for x in r["rids"].split(",") if x.strip()}
            for r in csv.DictReader(fh, delimiter="\t")
        }


def run(
    model: str | None = None,
    max_docs: int | None = None,
    embed_model: str | None = None,
    k: int = 5,
    think: bool = False,
    dataset: str = "t6",
) -> dict[str, Any]:
    """``dataset`` — t6(A, 30건 전부 군수) | t6b(A-2, 군수 15 / 부군수 15, 규칙을 만들 때 안 본 데이터)."""
    data_dir = track_data_dir(dataset)
    builder = OfficialLineBuilder(RULES, model=model, embed_model=embed_model, k=k, think=think)
    cfg, org = builder.cfg, builder.org
    gold = load_gold(data_dir / "approval_gold.tsv")
    row_gold = load_row_gold(data_dir / "row_gold.tsv")
    with (data_dir / "metadata.tsv").open(encoding="utf-8") as fh:
        meta = {r["no"].lstrip("0") or "0": r for r in csv.DictReader(fh, delimiter="\t")}

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for no in sorted(gold, key=int)[:max_docs]:
        g, m = gold[no], meta[no]
        doc = pymupdf.open(data_dir / m["저장파일명"])
        body = "\n".join(p.get_text() for p in doc)
        title, dept, date = m["제목"], g["org"], g["date"].replace(".", "-")
        path_gold = [c["title"] for c in g["cells"] if c["role"] != "협조"]
        gold_final = title_level(path_gold[-1])

        # 입력: 기안자 직위와 소속 팀(로그인 사용자). gold 에서 가져온다.
        drafter = path_gold[0]
        team_lead = path_gold[1] if title_level(drafter) == 0 and title_level(path_gold[1]) == 1 else None

        # 보조 지표: 담당업무로 팀을 맞힐 수 있나 (기안자를 모를 때의 라우팅)
        team_pred = team_gold = None
        teams = org.teams.get(dept, {})
        if team_lead and teams:
            team_gold = org.team_of_lead(dept, team_lead)
            names = list(teams)
            bm = BM25Okapi([tokenize(teams[t]["duties"]) for t in names])
            sc = bm.get_scores(tokenize(title + " " + body[:800]))
            team_pred = names[max(range(len(names)), key=lambda i: sc[i])]

        # 종점: 별표 조회
        t0 = time.perf_counter()
        final = builder.decide_final(dept, title, body)
        final.pop("candidates", None)
        pick_sec = round(time.perf_counter() - t0, 2)

        # 권한대행 기간엔 부군수 한 사람이 군수 권한과 부군수 권한을 다 행사한다 —
        # 칸만 봐서는 어느 레벨 결재였는지 못 가른다. 둘 다 정답으로 본다.
        acceptable = {gold_final}
        if any("권한대행" in c["note"] for c in g["cells"] if c["role"] == "결재"):
            acceptable = {4, 5}
        lvl = final["level"]
        verdict = ("unknown" if lvl is None else
                   "same" if lvl in acceptable else
                   "escalated" if lvl < min(acceptable) else "violation")

        chains = {
            # 경로 모드는 종점을 gold 로 준다(경로만 잰다). v1 은 전부 군수라 5 로 고정해도
            # 같았지만 v2 는 부군수 종점이 15건이다.
            "path_no_acting": build_slot_chain(dept, drafter, team_lead, org, cfg, date=date,
                                               final_level=gold_final, use_acting=False),
            "path_acting": build_slot_chain(dept, drafter, team_lead, org, cfg, date=date,
                                            final_level=gold_final),
            "agent": build_slot_chain(dept, drafter, team_lead, org, cfg, date=date,
                                      final_level=lvl if lvl is not None else 5),
        }
        scores = {mode: score([c["title"] for c in v], path_gold) for mode, v in chains.items()}

        # 업무항목(별표 행) 판정 — row_gold 에 라벨이 있는 문서만
        rg = row_gold.get(g["문서번호"])
        row_eval = None
        if rg and "rid" in final:
            row_eval = {"pick_ok": final["rid"] in rg,
                        "recall": bool(rg & set(final["candidate_rids"]))}

        cases.append({
            "문서번호": g["문서번호"], "dept": dept, "date": date, "row_eval": row_eval,
            "org_known": dept in org.head,
            "final": final, "final_gold_level": gold_final, "final_verdict": verdict,
            "final_gold_ambiguous": len(acceptable) > 1,
            "team_pred": team_pred, "team_gold": team_gold,
            "pred_path_acting": [c["title"] for c in chains["path_acting"]],
            "evidence": [c["evidence"] for c in chains["agent"]],
            "scores": scores, "pick_sec": pick_sec,
        })

    # 경로 점수는 조직도에 있는 부서만. 없는 부서는 국장 칸을 만들 근거가 없다.
    known = [c for c in cases if c["org_known"]]
    n = len(known) or 1

    def mean(mode: str, key: str) -> float:
        return round(sum(float(c["scores"][mode][key]) for c in known) / n, 4)

    team_cases = [c for c in cases if c["team_gold"]]
    row_cases = [c["row_eval"] for c in cases if c["row_eval"]]
    report = {
        "track": TRACK,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "org": "가평군",
        "model": model,
        "retrieval": f"hybrid(bm25+{embed_model})" if embed_model else "bm25",
        "k": k,
        "think": think,
        "rules": "rules/gapyeong.yaml + 별표1(official) + org_chart(official)",
        "method": "기안자 직위·팀은 입력. 조직도 슬롯 체인 → 직무대리 표 → 별표 조회로 종점",
        "dataset": dataset,
        "documents": len(cases),
        "path_docs": len(known),
        "summary": {
            mode: {"title_accuracy_mean": mean(mode, "title_accuracy"),
                   "exact": sum(c["scores"][mode]["exact"] for c in known),
                   "cell_count_match": sum(c["scores"][mode]["cell_count_match"] for c in known)}
            for mode in ("path_no_acting", "path_acting", "agent")
        },
        "final_verdicts": dict(Counter(c["final_verdict"] for c in cases)),
        "final_exact": sum(c["final_verdict"] == "same" for c in cases),
        "final_confusion": dict(Counter(
            f"{LEVEL_NAMES.get(c['final']['level'], '?')}→"
            f"{'부군수|군수' if c['final_gold_ambiguous'] else LEVEL_NAMES[c['final_gold_level']]}"
            for c in cases)),
        # 공개 원문은 부단체장 이상만 올라온다 → 별표가 국장 이하라고 한 문서는 전부 "상향"으로만
        # 관측된다. 채점이 성립하는 건 예측이 부군수·군수인 문서뿐이다.
        "final_decidable": {
            "n": sum((c["final"]["level"] or 0) >= 4 for c in cases),
            "correct": sum((c["final"]["level"] or 0) >= 4 and c["final_verdict"] == "same" for c in cases),
        },
        "final_sources": dict(Counter(c["final"]["source"] for c in cases)),
        "gold_final_levels": dict(Counter(LEVEL_NAMES[c["final_gold_level"]] for c in cases)),
        "row_classification": {
            "labeled": len(row_cases),
            "candidate_recall": sum(r["recall"] for r in row_cases),
            "pick_correct": sum(r["pick_ok"] for r in row_cases),
        },
        "team_routing": {"cases": len(team_cases),
                         "correct": sum(c["team_pred"] == c["team_gold"] for c in team_cases)},
        "cases": cases,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    results_dir = track_results_dir(TRACK)
    prefix = "" if dataset == "t6" else f"{dataset}-"
    out_path = results_dir / f"{prefix}{datetime.now():%Y%m%d-%H%M}.json"
    if out_path.exists():
        out_path = results_dir / f"{prefix}{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["results_path"] = str(out_path)
    return report


def report(rep: dict[str, Any]) -> None:
    """콘솔 출력. 이름은 애초에 gold 에 없다."""
    typer.echo(f"[t6] {rep['method']}")
    typer.echo(f"     가평군 {rep['dataset']} / 문서 {rep['documents']}건 / 모델 {rep['model'] or '(검색 1위)'}"
               f" / 검색 {rep['retrieval']} k={rep['k']}")
    typer.echo("")
    label = {"path_no_acting": "경로 (대리 표 없음, 종점=gold)",
             "path_acting": "경로 (대리 표 있음, 종점=gold)",
             "agent": "에이전트 (종점도 별표로)"}
    for mode, s in rep["summary"].items():
        typer.echo(f"  {label[mode]:30} 직위 {s['title_accuracy_mean']:6.1%}  "
                   f"완전일치 {s['exact']:2}/{rep['path_docs']}  칸수 {s['cell_count_match']:2}/{rep['path_docs']}")
    typer.echo("")
    typer.echo(f"  gold 종점 분포   {rep['gold_final_levels']}")
    typer.echo(f"  별표 vs 실제     {rep['final_verdicts']}   (escalated = 별표 최소보다 높이 결재)")
    typer.echo(f"  종점 근거        {rep['final_sources']}")
    typer.echo(f"  종점 일치        {rep['final_exact']}/{rep['documents']}   예측→실제 {rep['final_confusion']}")
    fd = rep["final_decidable"]
    typer.echo(f"  채점 가능 종점   {fd['correct']}/{fd['n']}   (예측이 부군수·군수인 문서 — 공개 원문이 부단체장 이상뿐이라)")
    if rep["path_docs"] < rep["documents"]:
        typer.echo(f"  (경로 점수는 조직도에 있는 부서 {rep['path_docs']}건만)")
    rc = rep["row_classification"]
    typer.echo(f"  별표 행 판정     후보 재현 {rc['candidate_recall']}/{rc['labeled']}  "
               f"선택 정답 {rc['pick_correct']}/{rc['labeled']}   (row_gold 라벨 문서만)")
    tr = rep["team_routing"]
    typer.echo(f"  담당업무→팀      {tr['correct']}/{tr['cases']}")
    typer.echo("")
    for c in rep["cases"]:
        f = c["final"]
        lvl = LEVEL_NAMES.get(f["level"], "?") if f["level"] is not None else "?"
        amt = f"{f['amount']:,}원({f['amount_label']})" if f["amount"] else ""
        row = (f["row"] or f"{f['category']} 금액 미상")[:40]
        ok = "".join("O" if c["scores"][k]["exact"] else "." for k in ("path_no_acting", "path_acting", "agent"))
        typer.echo(f"  {c['문서번호']:22} {ok}  {lvl:3} {c['final_verdict']:9} {amt:26} {row}")
    typer.echo("")
    typer.echo(f"  → {rep['results_path']}  ({rep['elapsed_sec']}s)")
