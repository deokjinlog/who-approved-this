"""T1 — 공공 결재문서 필드·결재선 추출 (본 과제와 가장 가까운 트랙).

목적
    결재문서 PDF에서 메타데이터와 결재선을 뽑는 정확도를 숫자로 낸다.
    T4가 "글자를 얼마나 잘 읽나"였다면 여기는 "필드를 제대로 뽑았나"다.

T1a / T1b 를 가르는 이유
    포털 메타데이터에 **결재선이 없다.** ``결재정보`` 컬럼은 10건 전부
    "(화면에 표시 항목 없음)" 이고 실명은 담당자(기안자)까지만 나온다.

    * **T1a** — 6개 필드(제목·생산기관·담당부서·담당자·생산일자·문서번호). 정답 있음.
    * **T1b** — 결재선(기안/검토/결재/협조). **정답 없음.** 텍스트 레이어 추출값을
      잠정 기준으로 두고 나중에 OCR·VLM 값과 대조한다.

개인정보
    결과 JSON에는 문서 id(저장파일명 stem), 필드별 일치 bool, 결재선의 역할별 개수와
    직위 목록만 남긴다. **제목·이름 같은 값은 저장하지 않는다.**
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from who_approved_this.collect.opengokr_local import OpenGoKrLocalCollector
from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.evaluate.approval_match import (
    ROLES as GOLD_ROLES,
    ApprovalMatchEvaluator,
    load_gold,
)
from who_approved_this.evaluate.field_match import FIELDS, ROLES, FieldMatchEvaluator
from who_approved_this.parse.approval_ocr import ApprovalLineParser
from who_approved_this.parse.fields_textlayer import FieldTextLayerParser

TRACK = "t1"


def _run_t1b(
    data_dir: Path, sources: tuple[str, ...], max_docs: int | None
) -> dict[str, Any]:
    """T1b — 결재선 추출을 source 별로 돌려 gold 와 대조한다."""
    gold_path = data_dir / "approval_gold.tsv"
    if not gold_path.is_file():
        return {"skipped": f"{gold_path.name} 없음"}

    gold = load_gold(gold_path)
    collector = OpenGoKrLocalCollector(data_dir, max_docs=max_docs)
    out: dict[str, Any] = {"gold_docs": len(gold), "sources": {}}

    for source in sources:
        evaluator = ApprovalMatchEvaluator(gold)
        parser = ApprovalLineParser(data_dir / "crops", source=source)
        docs: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for pdf_path, truth in collector.items():
            parsed = parser.parse(pdf_path)
            scores = evaluator.evaluate(parsed, truth)
            docs.append(
                {
                    "doc_id": truth["doc_id"],
                    "region": parsed["region"],
                    # 값은 담지 않는다. 개수와 일치 여부만.
                    "cells": {
                        r: {
                            "gold": int(scores[f"role_{r}_gold_cells"]),
                            "pred": int(scores[f"role_{r}_pred_cells"]),
                            "count_match": bool(scores[f"role_{r}_count_match"]),
                            "title_hits": int(scores[f"role_{r}_title_hits"]),
                            "name_hits": int(scores[f"role_{r}_name_hits"]),
                            "name_scored": int(scores[f"role_{r}_name_scored"]),
                        }
                        for r in GOLD_ROLES
                    },
                    "title_accuracy": round(scores["title_accuracy"], 4),
                    "name_accuracy": round(scores["name_accuracy"], 4),
                    "cell_count_match": round(scores["cell_count_match"], 4),
                    "unreadable_cells": int(scores["unreadable_cells"]),
                    "failures": [{"role": r, "reason": why} for r, why in evaluator.failures],
                }
            )
        n = len(docs) or 1
        out["sources"][source] = {
            "parser": parser.name,
            "documents": docs,
            "summary": {
                "documents": len(docs),
                "title_accuracy_mean": round(sum(d["title_accuracy"] for d in docs) / n, 4),
                "name_accuracy_mean": round(sum(d["name_accuracy"] for d in docs) / n, 4),
                "cell_count_match_mean": round(sum(d["cell_count_match"] for d in docs) / n, 4),
                "docs_all_cells_matched": sum(
                    1 for d in docs if d["cell_count_match"] == 1.0
                ),
                "unreadable_cells": sum(d["unreadable_cells"] for d in docs),
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            },
        }
    return out


def run(max_docs: int | None = None) -> dict[str, Any]:
    """T1을 collect → parse → evaluate 순서로 돌리고 결과 JSON을 남긴다."""
    data_dir = track_data_dir(TRACK)
    results_dir = track_results_dir(TRACK)

    collector = OpenGoKrLocalCollector(data_dir, max_docs=max_docs)
    parser = FieldTextLayerParser()
    evaluator = FieldMatchEvaluator()

    started = time.perf_counter()
    documents: list[dict[str, Any]] = []
    for pdf_path, truth in collector.items():
        parsed = parser.parse(pdf_path)
        scores = evaluator.evaluate(parsed, truth)
        line = parsed["approval_line"]
        documents.append(
            {
                "doc_id": truth["doc_id"],
                # 값은 담지 않는다. 일치 여부와 구조만.
                "fields": {f: bool(scores[f"{f}_norm"]) for f in FIELDS},
                "fields_exact": {f: bool(scores[f"{f}_exact"]) for f in FIELDS},
                "field_accuracy": round(scores["field_accuracy"], 4),
                "failures": [{"field": f, "reason": r} for f, r in evaluator.failures],
                "approval_line": {
                    "count": len(line),
                    "roles": {r: int(scores[f"approval_{r}_count"]) for r in ROLES},
                    "titles": [c["title"] for c in line],
                    "has_text": bool(line),
                },
            }
        )
    elapsed = time.perf_counter() - started

    if not documents:
        raise FileNotFoundError(f"{data_dir} 에 채점할 문서가 없다. 건너뜀: {collector.skipped}")

    t1b = _run_t1b(data_dir, ("textlayer", "ocr"), max_docs)

    now = datetime.now()
    report = {
        "track": TRACK,
        "run_at": now.isoformat(timespec="seconds"),
        "parser": parser.name,
        "evaluator": evaluator.name,
        "ground_truth": "open.go.kr metadata.tsv (T1a) / 정답 없음 (T1b)",
        "note": "결과에는 값(제목·이름)을 저장하지 않는다. 일치 여부·개수·직위만.",
        "documents": documents,
        "summary": {
            "documents": len(documents),
            "field_accuracy_mean": round(
                sum(d["field_accuracy"] for d in documents) / len(documents), 4
            ),
            "by_field": {
                f: round(sum(d["fields"][f] for d in documents) / len(documents), 4)
                for f in FIELDS
            },
            "approval_line_text_docs": sum(
                1 for d in documents if d["approval_line"]["has_text"]
            ),
            "title_histogram": dict(
                Counter(t for d in documents for t in d["approval_line"]["titles"])
            ),
            "elapsed_sec": round(elapsed, 2),
        },
        "t1b": t1b,
        "skipped": [{"file": f, "reason": r} for f, r in collector.skipped],
    }

    out_path = results_dir / f"{now:%Y%m%d-%H%M}.json"
    if out_path.exists():
        out_path = results_dir / f"{now:%Y%m%d-%H%M%S}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["results_path"] = str(out_path)
    return report


def report(rep: dict[str, Any]) -> None:
    """콘솔 출력. 실패는 **문서 id / 필드 / 사유** 로만 — 값은 찍지 않는다."""
    s = rep["summary"]
    typer.echo(f"[t1] {rep['parser']} / {rep['evaluator']}")
    typer.echo(f"     정답: {rep['ground_truth']}")
    typer.echo("")
    typer.echo("  T1a 필드별 일치율 (정규화 후)")
    for f, v in s["by_field"].items():
        bar = "█" * int(v * 20)
        typer.echo(f"    {f:6} {v:6.1%} {bar}")
    typer.echo(f"    = 문서 평균 {s['field_accuracy_mean']:.1%}")
    typer.echo("")
    typer.echo(
        f"  T1b 결재선 텍스트 추출: {s['approval_line_text_docs']}/{s['documents']}건"
        f"  직위 분포 {s['title_histogram']}"
    )
    typer.echo("")
    typer.echo("  실패 목록 (값 미출력 — PDF를 직접 열어 확인)")
    for d in rep["documents"]:
        for fail in d["failures"]:
            typer.echo(f"    {d['doc_id'][:34]:36} {fail['field']:6} {fail['reason']}")
    b = rep.get("t1b", {})
    if "sources" in b:
        typer.echo("")
        typer.echo(f"  T1b 결재선 — gold {b['gold_docs']}건 (사람이 직접 만든 정답)")
        typer.echo(f"    {'source':10} {'직위':>7} {'이름':>7} {'칸수':>7}  칸수전부맞은문서")
        for src, data in b["sources"].items():
            m = data["summary"]
            typer.echo(
                f"    {src:10} {m['title_accuracy_mean']:6.1%} {m['name_accuracy_mean']:6.1%}"
                f" {m['cell_count_match_mean']:6.1%}   {m['docs_all_cells_matched']}/{m['documents']}"
                f"   ({m['elapsed_sec']}s)"
            )
        typer.echo(f"    이름 미판독(gold '?') 칸: {list(b['sources'].values())[0]['summary']['unreadable_cells']}개 — 채점 제외")
        typer.echo("")
        typer.echo("  T1b 실패 (값 미출력)")
        for src, data in b["sources"].items():
            for d in data["documents"]:
                for fail in d["failures"][:4]:
                    typer.echo(f"    [{src:9}] {d['doc_id'][:30]:32} {fail['role']:4} {fail['reason']}")

    typer.echo("")
    typer.echo(f"  → {rep['results_path']}  ({s['elapsed_sec']}s)")
