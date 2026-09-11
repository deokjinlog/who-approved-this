"""T1s — 결재문서 본문 요약 (첨부 없이). 회사 기능 ③ 의 본문 부분.

    입력    T1 결재문서 본문(텍스트 레이어, 쪽별)
    요약    templates/summary.yaml v2 슬롯. 여러 쪽 문서는 통째(whole) / 쪽별 병합(paged) 둘 다
    검증    evaluate/summary_check — 환각률(필터 전·후), 실명 0, 원문 위치(쪽), 길이 비율
    사람    DATA_ROOT/t1/summary_review/ 에 (원문 첫 200자 / 요약) 쌍 10건

결과 JSON 에는 **개수와 비율만** 남긴다(요약 문장·업체명·원문은 리포에 안 들어간다).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
import typer

from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.evaluate.summary_check import check
from who_approved_this.parse.summarizer import load_summary_template, render_markdown, summarize_doc
from who_approved_this.tracks.t1c_approval_predict import all_names, load_docs

TRACK = "t1s"
TEMPLATE = Path(__file__).resolve().parents[3] / "templates" / "summary.yaml"
REVIEW_N = 10


def _pages(pdf: Path) -> list[str]:
    with pymupdf.open(pdf) as doc:
        return [doc[i].get_text(sort=True) for i in range(doc.page_count)]


def run(model: str | None = None, max_docs: int | None = None) -> dict[str, Any]:
    if not model:
        raise typer.BadParameter("t1s 는 --model 이 필요하다 (생성 칸)")
    data_dir = track_data_dir("t1")
    template = load_summary_template(TEMPLATE)
    docs = load_docs(data_dir)[:max_docs]
    names = all_names(docs, data_dir)
    # 이름 → 직위 (같은 조직 gold 에서). 모르면 ○○○
    name_to_title = {c["name"]: c["title"] for d in docs for c in d["cells"]
                     if c["name"] not in ("", "-", "?") and c["title"] not in ("", "-")}
    review_dir = data_dir / "summary_review"
    review_dir.mkdir(exist_ok=True)

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for i, d in enumerate(docs):
        pages = _pages(d["pdf"])
        modes = ["whole", "paged"] if len(pages) > 1 else ["whole"]
        for mode in modes:
            res = summarize_doc(model, pages, template, name_to_title, names, mode=mode)
            chk = check(res, [p for p in pages], names)
            cases.append({"doc_id": d["doc_id"], "pages": len(pages), "mode": mode,
                          "checks": res["checks"], "eval": chk,
                          "vendors_found": len(res["slots"]["vendors"])})
            if i < REVIEW_N or len(pages) > 1:
                md = render_markdown(res, d.get("문서번호", d["doc_id"]))
                head = "\n".join(pages)[:200]
                (review_dir / f"{d['doc_id']}.{mode}.md").write_text(
                    f"# 원문 첫 200자\n\n```\n{head}\n```\n\n{md}", encoding="utf-8")

    def agg(sel: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sel) or 1
        tok = sum(c["eval"]["hallucination_raw"]["tokens"] for c in sel)
        bad = sum(c["eval"]["hallucination_raw"]["ungrounded"] for c in sel)
        ftok = sum(c["eval"]["hallucination_final"]["tokens"] for c in sel)
        fbad = sum(c["eval"]["hallucination_final"]["ungrounded"] for c in sel)
        loc_c = sum(c["eval"]["location"]["checked"] for c in sel)
        loc_ok = sum(c["eval"]["location"]["correct"] for c in sel)
        return {
            "cases": len(sel),
            "json_ok": sum(c["checks"]["json_ok"] for c in sel),
            "hallucination_raw_rate": round(bad / tok, 4) if tok else 0.0,
            "hallucination_raw": f"{bad}/{tok}",
            "hallucination_final_rate": round(fbad / ftok, 4) if ftok else 0.0,
            "dropped_items": sum(c["checks"]["dropped_ungrounded_number"] + c["checks"]["dropped_invented_ref"] for c in sel),
            "names_in_output": sum(c["eval"]["names_in_output"] for c in sel),
            "location": f"{loc_ok}/{loc_c}",
            "length_ratio_mean": round(sum(c["eval"]["length_ratio"] for c in sel) / n, 4),
            "sec_mean": round(sum(c["checks"]["sec"] for c in sel) / n, 2),
            "llm_calls": sum(c["checks"]["llm_calls"] for c in sel),
        }

    whole = [c for c in cases if c["mode"] == "whole"]
    multi = sorted({c["doc_id"] for c in cases if c["pages"] > 1})
    report = {
        "track": TRACK, "run_at": datetime.now().isoformat(timespec="seconds"), "model": model,
        "template": "templates/summary.yaml v2", "documents": len(docs),
        "note": "첨부 없이 본문만. 결과엔 개수·비율만 — 요약 문장·업체명은 DATA_ROOT/t1/summary_review/",
        "summary_whole_all": agg(whole),
        "multi_page": {doc: {m: agg([c for c in cases if c["doc_id"] == doc and c["mode"] == m])
                             for m in ("whole", "paged")} for doc in multi},
        "review_dir": str(review_dir),
        "cases": cases,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    out = track_results_dir(TRACK) / f"{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["results_path"] = str(out)
    return report


def report(rep: dict[str, Any]) -> None:
    typer.echo(f"[t1s] 본문 요약 {rep['documents']}건 / {rep['model']} / {rep['template']}")
    s = rep["summary_whole_all"]
    typer.echo(f"  통째(whole) 전체: JSON {s['json_ok']}/{s['cases']} | 환각률(필터 전) {s['hallucination_raw_rate']:.1%}"
               f" ({s['hallucination_raw']}) → 필터 후 {s['hallucination_final_rate']:.1%} (제외 {s['dropped_items']}항목)"
               f" | 실명 {s['names_in_output']} | 길이비 {s['length_ratio_mean']:.3f} | {s['sec_mean']}s/건")
    for doc, m in rep["multi_page"].items():
        for mode, a in m.items():
            typer.echo(f"  {doc[-24:]:24} {mode:5} 환각률 {a['hallucination_raw_rate']:.1%} ({a['hallucination_raw']})"
                       f" | 위치 {a['location']} | 길이비 {a['length_ratio_mean']:.3f} | LLM {a['llm_calls']}회 {a['sec_mean']}s")
    typer.echo(f"  검토쌍 → {rep['review_dir']}")
    typer.echo(f"  → {rep['results_path']}  ({rep['elapsed_sec']}s)")
