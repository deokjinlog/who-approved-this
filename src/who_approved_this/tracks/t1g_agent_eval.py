"""T1g — 전자결재 에이전트 통합 평가.

T1f 가 결재선만 쟀다면, 여기는 에이전트 한 번 호출이 내는 **전부**를 같은 조건으로 잰다.

조건 — 새 문서를 기안하는 상황을 흉내 낸다
    입력: 기안자 조직·직위(gold 첫 칸), 문서 제목. **본문·첨부는 주지 않는다.**
    대상 문서는 참고에서 뺀다(leave-one-out).

재는 것
    업무항목   worktype 라벨과 일치
    결재선     gold 와 직위·이름·칸수 (T1f 와 같은 대조기)
    초안       칸 상태 분포(채움/생성/입력필요/실패), 지어낸 문서번호·법령명·다른 기관명,
               실제 문서 대비 핵심어 재현율
    확인 필요  에이전트가 스스로 올린 항목 수
    시간       단계별

비교 기준은 T1d 의 자유 생성(인용 문서번호 7개 중 6개 허구, 다른 기관명 오염 3/24)이다.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from typing import Any

import typer

from who_approved_this.agent import ApprovalAgent
from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.evaluate.approval_match import ApprovalMatchEvaluator, load_gold
from who_approved_this.tracks.t1c_approval_predict import load_docs
from who_approved_this.tracks.t1d_draft import score_draft
from who_approved_this.tracks.t1e_worktype import load_rules as load_worktypes

TRACK = "t1g"


def run(model: str | None = None, org: str = "경기도서관", max_docs: int | None = None) -> dict[str, Any]:
    """에이전트를 대상 조직 문서마다 한 번씩 돌려 결과를 잰다."""
    data_dir = track_data_dir("t1")
    docs = load_docs(data_dir)
    agent = ApprovalAgent(model, docs=docs)
    evaluator = ApprovalMatchEvaluator(load_gold(data_dir / "approval_gold.tsv"))
    labels = load_worktypes()["labels"]

    targets = [d for d in docs if d["org"] == org][:max_docs]
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for d in targets:
        drafter = next((c["title"] for c in d["cells"] if c["role"] == "기안"), "주무관")
        result = agent.handle(
            {"org": org, "title_of_user": drafter, "title": d["제목"], "exclude_doc_id": d["doc_id"]}
        )
        line = result["approval_line"]["cells"]
        scores = evaluator.evaluate({"cells": line}, d)
        draft = result["draft"]
        quality = score_draft(d["body"], draft["markdown"])
        statuses = Counter(s["status"] for s in draft["sections"])
        cases.append(
            {
                "doc_id": d["doc_id"],
                "work_type": result["work_type"]["value"],
                "work_type_ok": result["work_type"]["value"] == labels.get(d.get("문서번호", "")),
                "title_accuracy": round(scores["title_accuracy"], 4),
                "name_accuracy": round(scores["name_accuracy"], 4),
                "cell_count_match": round(scores["cell_count_match"], 4),
                "draft_template": draft["template"]["work_type"],
                "draft_status": dict(statuses),
                "draft_checks": draft["checks"],
                "keyphrases_found": quality["keyphrases_found"],
                "keyphrases_expected": quality["keyphrases_expected"],
                "needs_confirmation": len(result["needs_confirmation"]),
                "timings": result["timings"],
            }
        )

    n = len(cases) or 1
    total = lambda key: sum(c["draft_checks"].get(key, 0) for c in cases)
    status_total: Counter[str] = Counter()
    for c in cases:
        status_total.update(c["draft_status"])
    step_mean = {
        step: round(sum(c["timings"].get(step, 0.0) for c in cases) / n, 2)
        for step in ("identify", "classify", "approval_line", "draft")
    }
    report = {
        "track": TRACK,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "org": org,
        "model": model,
        "condition": "제목만 입력(본문·첨부 없음), 기안자 조직·직위는 입력, leave-one-out",
        "documents": len(cases),
        "summary": {
            "work_type_accuracy": round(sum(c["work_type_ok"] for c in cases) / n, 4),
            "title_accuracy_mean": round(sum(c["title_accuracy"] for c in cases) / n, 4),
            "name_accuracy_mean": round(sum(c["name_accuracy"] for c in cases) / n, 4),
            "cell_count_match_mean": round(sum(c["cell_count_match"] for c in cases) / n, 4),
            "draft_status_total": dict(status_total),
            "invented_refs": total("invented_refs"),
            "invented_laws": total("invented_laws"),
            "org_contamination": total("org_contamination"),
            "keyphrase_recall": round(
                sum(c["keyphrases_found"] for c in cases)
                / max(sum(c["keyphrases_expected"] for c in cases), 1), 4),
            "needs_confirmation_mean": round(sum(c["needs_confirmation"] for c in cases) / n, 2),
            "sec_per_step": step_mean,
        },
        "baseline_t1d_free_generation": {
            "invented_refs": "인용 7개 중 6개 허구", "org_contamination": "3/24",
        },
        "cases": cases,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    out = track_results_dir(TRACK) / f"{datetime.now():%Y%m%d-%H%M}.json"
    if out.exists():
        out = track_results_dir(TRACK) / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["results_path"] = str(out)
    return report


def report(rep: dict[str, Any]) -> None:
    """콘솔 출력. 이름·본문은 찍지 않는다."""
    s = rep["summary"]
    typer.echo(f"[t1g] 전자결재 에이전트 통합 평가 — {rep['org']} {rep['documents']}건 / {rep['model']}")
    typer.echo(f"      조건: {rep['condition']}")
    typer.echo("")
    typer.echo(f"  업무항목 정확도      {s['work_type_accuracy']:.1%}")
    typer.echo(f"  결재선 직위 / 이름   {s['title_accuracy_mean']:.1%} / {s['name_accuracy_mean']:.1%}"
               f"   칸수 {s['cell_count_match_mean']:.1%}")
    typer.echo(f"  초안 칸 상태         {s['draft_status_total']}")
    typer.echo(f"  지어낸 문서번호      {s['invented_refs']}  (걸러낸 수)")
    typer.echo(f"  지어낸 법령명        {s['invented_laws']}")
    typer.echo(f"  다른 기관명 오염     {s['org_contamination']}")
    typer.echo(f"  핵심어 재현율        {s['keyphrase_recall']:.1%}")
    typer.echo(f"  확인 필요 (평균)     {s['needs_confirmation_mean']}건")
    typer.echo(f"  단계별 시간(평균)    {s['sec_per_step']}")
    typer.echo("")
    typer.echo(f"  → {rep['results_path']}  ({rep['elapsed_sec']}s)")
