"""T1f — 결재선 자동 지정 에이전트 평가.

기안자 직위 + 문서를 주고 결재선을 **만들어** gold 와 대조한다.
비교군은 T1c 의 이력 기반 예측기(다수결 + LLM 이름)다.

이력 기반과 규칙 기반은 필요한 게 다르다.
    이력 기반   같은 조직 과거 문서 5~7건
    규칙 기반   전결표 + 조직도 (과거 문서 0건이어도 동작)

신설 부서·새 문서 유형에서는 이력이 없으므로 규칙 기반만 답을 낼 수 있다.
반대로 규칙이 없으면 이력이 유일한 근거다. 둘은 대체재가 아니라 보완재다.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.evaluate.approval_match import (
    ApprovalMatchEvaluator,
    load_gold,
    normalize_token,
)
from who_approved_this.parse.approval_agent import build_line, load_roster, load_rules
from who_approved_this.tracks.t1c_approval_predict import load_docs
from who_approved_this.tracks.t1c_predictors import _THINK_BLOCK, ollama_chat
from who_approved_this.tracks.t1e_worktype import (
    SYSTEM as WT_SYSTEM,
    build_prompt as wt_prompt,
    classify_by_title,
    load_rules as load_worktype,
)

TRACK = "t1f"
RULES = Path(__file__).resolve().parents[3] / "rules" / "approval.yaml"


def run(
    model: str | None = None,
    org: str = "경기도서관",
    max_docs: int | None = None,
) -> dict[str, Any]:
    """T1f를 돌린다. ``model`` 이 있으면 업무항목을 LLM 이 분류한다."""
    data_dir = track_data_dir("t1")
    rules = load_rules(RULES)
    roster = load_roster(Path(rules["roster"]["path"]))
    wt = load_worktype()
    types, labels = wt["types"], wt["labels"]
    valid = {t["key"] for t in types}
    evaluator = ApprovalMatchEvaluator(load_gold(data_dir / "approval_gold.tsv"))

    docs = [d for d in load_docs(data_dir) if d["org"] == org][:max_docs]
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []

    for d in docs:
        gold_cells = d["cells"]
        drafter = next(
            (c["title"] for c in gold_cells if c["role"] == "기안"), "주무관"
        )

        # ① 업무항목
        t0 = time.perf_counter()
        if model:
            raw = ollama_chat(model, WT_SYSTEM, wt_prompt(d["body"], types), 0.0, 300)
            answer = _THINK_BLOCK.sub("", raw or "").strip().splitlines()
            picked = answer[0].strip().strip('"') if answer else ""
            work_type = picked if picked in valid else classify_by_title(d["제목"])
            wt_source = "llm" if picked in valid else "title-rule"
        else:
            work_type = classify_by_title(d["제목"])
            wt_source = "title-rule"
        wt_sec = round(time.perf_counter() - t0, 2)

        # ②③④
        out = build_line(org, drafter, work_type, d["제목"], d["body"], rules, roster)
        scores = evaluator.evaluate({"cells": out["cells"]}, d)

        cases.append(
            {
                "doc_id": d["doc_id"],
                "work_type": work_type,
                "work_type_gold": labels.get(d.get("문서번호", ""), ""),
                "work_type_ok": work_type == labels.get(d.get("문서번호", ""), ""),
                "work_type_source": wt_source,
                "amount": out["amount"],
                "team": out["team"],
                "final_title": out["final_title"],
                "title_accuracy": round(scores["title_accuracy"], 4),
                "name_accuracy": round(scores["name_accuracy"], 4),
                "cell_count_match": round(scores["cell_count_match"], 4),
                "evidence": dict(Counter(c["evidence"].split(":")[0] for c in out["cells"])),
                "elapsed_sec": wt_sec,
            }
        )

    n = len(cases) or 1
    report = {
        "track": TRACK,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "org": org,
        "model": model,
        "rules": "rules/approval.yaml",
        "method": "기안자 직위는 입력. ①업무항목(LLM) → ②금액 → ③전결표 → ④조직도 경로",
        "documents": len(cases),
        "summary": {
            "work_type_accuracy": round(sum(c["work_type_ok"] for c in cases) / n, 4),
            "title_accuracy_mean": round(sum(c["title_accuracy"] for c in cases) / n, 4),
            "name_accuracy_mean": round(sum(c["name_accuracy"] for c in cases) / n, 4),
            "cell_count_match_mean": round(sum(c["cell_count_match"] for c in cases) / n, 4),
            "cases_title_perfect": sum(1 for c in cases if c["title_accuracy"] == 1.0),
            "sec_per_case": round(sum(c["elapsed_sec"] for c in cases) / n, 2),
        },
        "evidence_totals": dict(
            Counter(k for c in cases for k, v in c["evidence"].items() for _ in range(v))
        ),
        "cases": cases,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    results_dir = track_results_dir(TRACK)
    out_path = results_dir / f"{datetime.now():%Y%m%d-%H%M}.json"
    if out_path.exists():
        out_path = results_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["results_path"] = str(out_path)
    return report


def report(rep: dict[str, Any]) -> None:
    """콘솔 출력. 이름은 찍지 않는다."""
    s = rep["summary"]
    typer.echo(f"[t1f] {rep['method']}")
    typer.echo(f"      조직 {rep['org']} / 문서 {rep['documents']}건 / 모델 {rep['model'] or '(제목규칙)'}")
    typer.echo("")
    typer.echo(f"  ① 업무항목 분류 정확도  {s['work_type_accuracy']:.1%}")
    typer.echo(f"  결재선 직위 일치        {s['title_accuracy_mean']:.1%}")
    typer.echo(f"  결재선 이름 일치        {s['name_accuracy_mean']:.1%}")
    typer.echo(f"  칸 수 일치              {s['cell_count_match_mean']:.1%}")
    typer.echo(f"  직위 전부 맞은 문서     {s['cases_title_perfect']}/{rep['documents']}")
    typer.echo(f"  근거 분포               {rep['evidence_totals']}")
    typer.echo("")
    typer.echo(f"  {'문서':22} {'업무항목':12} {'팀':14} {'직위':>6} {'이름':>6}")
    for c in rep["cases"]:
        mark = "" if c["work_type_ok"] else f" (gold={c['work_type_gold']})"
        typer.echo(
            f"  {c['doc_id'][-12:]:22} {c['work_type']:12} {str(c['team'] or '-'):14}"
            f" {c['title_accuracy']:6.2f} {c['name_accuracy']:6.2f}{mark}"
        )
    typer.echo("")
    typer.echo(f"  → {rep['results_path']}  ({rep['elapsed_sec']}s)")
