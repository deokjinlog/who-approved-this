"""T5 — 관보 안건 요약·검색 (RAG 가 관보에서도 동작하는가).

T1d 가 "같은 조직 결재문서로 초안을 쓴다"였다면, T5 는 같은 파이프라인을
**관보**에 그대로 얹어 본다. 문서 성격이 완전히 달라도(법령 개정문, 고시·공고)
검색·요약이 작동하는지, 어디서 무너지는지를 같은 지표로 본다.

정답을 어디서 얻나
    관보 1~2페이지 **목차**가 그 호의 안건을 모두 나열하고 각 항목에 **공식 제명과
    시작 페이지**가 붙어 있다. 시작 페이지로 본문을 자르면
    ``(본문, 공식 제명)`` 쌍이 자동으로 만들어진다. 사람이 라벨을 달 필요가 없다.

두 단계
    1. **검색** — 제명으로 질의해 그 안건 본문이 상위에 오는가(bm25/vector/hybrid).
    2. **요약** — 안건 본문을 요약하고 제명·본문 근거와 대조.

개인정보
    관보는 공개 문서이고 등장 인물은 공포문의 공직자다. 그래도 결과 JSON 에는
    제명·본문을 담지 않고 **점수만** 남긴다.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from datetime import datetime
from typing import Any

import typer

from who_approved_this.collect.gazette_toc import load_items
from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.evaluate.summary_match import SummaryMatchEvaluator
from who_approved_this.tracks.t1c_predictors import _THINK_BLOCK, ollama_generate
from who_approved_this.tracks.t1d_retrieval import (
    BM25Retriever,
    HybridRetriever,
    VectorRetriever,
)
from who_approved_this.tracks.t1d_draft import ollama_embed

TRACK = "t5"

#: 요약에 넣을 본문 최대 길이. 안건 하나가 수십만 자인 경우가 있다(첨부 표 등).
BODY_LIMIT = 4000

#: 본문이 이보다 짧으면 요약 대상에서 뺀다(제명만 있고 내용이 없는 정정 공고 등).
MIN_BODY = 300

#: 요약 표본 크기. 268건 전부 LLM 을 돌리면 오래 걸려, 종류별로 고르게 뽑는다.
SAMPLE_N = 24

SYSTEM = """너는 한국 행정문서 요약자다. 관보에 실린 안건 하나를 요약한다.

규칙:
- 본문에 있는 내용만 쓴다. 조문 번호·날짜·금액을 지어내지 않는다.
- 무엇이 어떻게 바뀌는지를 먼저 쓴다.
- 3문장 이내로 쓴다.

출력은 요약문만. 머리말이나 설명을 붙이지 않는다."""


def _sample(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """종류(법률/대통령령/고시/공고…)별로 고르게 뽑는다."""
    usable = [i for i in items if MIN_BODY <= i["body_chars"]]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in usable:
        by_kind.setdefault(item["kind"], []).append(item)
    rng = random.Random(20260909)
    for group in by_kind.values():
        rng.shuffle(group)

    picked: list[dict[str, Any]] = []
    kinds = sorted(by_kind)
    while len(picked) < n and any(by_kind[k] for k in kinds):
        for k in kinds:
            if by_kind[k] and len(picked) < n:
                picked.append(by_kind[k].pop())
    return picked


def _chunks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """검색용 청크. 안건 하나를 청크 하나로 둔다(경계가 이미 명확하다)."""
    return [
        {
            "chunk_id": i["item_id"],
            "doc_id": i["item_id"],
            "org": i["source"],
            "doc_type": i["kind"],
            "text": i["body"][:BODY_LIMIT],
        }
        for i in items
        if i["body_chars"] >= MIN_BODY
    ]


def run(
    model: str | None = None,
    embed_model: str | None = None,
    sample: int = SAMPLE_N,
) -> dict[str, Any]:
    """T5를 돌린다. ``model`` 이 없으면 검색까지만."""
    data_dir = track_data_dir("t4")  # 관보는 T4 가 쓰는 디렉터리에 있다
    results_dir = track_results_dir(TRACK)

    items: list[dict[str, Any]] = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        items += load_items(pdf)
    if not items:
        raise FileNotFoundError(f"{data_dir} 에서 관보 안건을 찾지 못했다")

    chunks = _chunks(items)
    bm25 = BM25Retriever(chunks)
    retrievers: list[Any] = [bm25]
    if embed_model:
        vector = VectorRetriever(chunks, ollama_embed(embed_model))
        retrievers += [vector, HybridRetriever(bm25, vector)]

    targets = _sample(items, sample)
    started = time.perf_counter()

    retrieval: dict[str, Any] = {}
    for retriever in retrievers:
        t0 = time.perf_counter()
        hits_at1 = hits_at3 = same_kind3 = 0
        for item in targets:
            found = retriever.search(item["title"], k=3)
            ids = [c["chunk_id"] for c, _ in found]
            hits_at1 += int(bool(ids) and ids[0] == item["item_id"])
            hits_at3 += int(item["item_id"] in ids)
            same_kind3 += int(any(c["doc_type"] == item["kind"] for c, _ in found))
        n = len(targets) or 1
        retrieval[retriever.name] = {
            "cases": len(targets),
            "self_at1": round(hits_at1 / n, 3),
            "self_at3": round(hits_at3 / n, 3),
            "same_kind_at3": round(same_kind3 / n, 3),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    summary_block: dict[str, Any] = {"skipped": "모델 미지정"}
    if model:
        summary_block = _summarize_all(model, targets)

    now = datetime.now()
    report = {
        "track": TRACK,
        "run_at": now.isoformat(timespec="seconds"),
        "source": "관보 (gwanbo.go.kr) — 목차가 정답 제명과 시작 페이지를 준다",
        "issues": sorted({i["source"] for i in items}),
        "items_total": len(items),
        "items_indexed": len(chunks),
        "kinds": dict(Counter(i["kind"] for i in items)),
        "sample": len(targets),
        "embed_model": embed_model,
        "model": model,
        "retrieval": retrieval,
        "summary": summary_block,
        "note": "결과에는 제명·본문을 담지 않는다. 점수만.",
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


def _summarize_all(model: str, targets: list[dict[str, Any]]) -> dict[str, Any]:
    """표본 안건을 요약하고 제명·본문 근거와 대조한다."""
    evaluator = SummaryMatchEvaluator()
    cases: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for item in targets:
        body = item["body"][:BODY_LIMIT]
        prompt = f"{SYSTEM}\n\n# 관보 안건 본문\n\n{body}\n\n위 안건을 요약하라."
        raw = ollama_generate(model, prompt, temperature=0.2, timeout=300) or ""
        text = _THINK_BLOCK.sub("", raw).strip()
        scores = evaluator.evaluate(text, {"title": item["title"], "body": body})
        cases.append(
            {
                "item_id": item["item_id"],
                "kind": item["kind"],
                "body_chars": item["body_chars"],
                **{k: round(v, 4) for k, v in scores.items()},
                "empty": not text,
            }
        )
    n = len(cases) or 1
    return {
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "title_recall_mean": round(sum(c["title_recall"] for c in cases) / n, 4),
            "grounded_ratio_mean": round(sum(c["grounded_ratio"] for c in cases) / n, 4),
            "numbers_total": int(sum(c["numbers_total"] for c in cases)),
            "numbers_grounded": int(sum(c["numbers_grounded"] for c in cases)),
            "compression_mean": round(sum(c["compression"] for c in cases) / n, 4),
            "empty_outputs": sum(1 for c in cases if c["empty"]),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
        "by_kind": {
            k: {
                "cases": len(rows),
                "title_recall": round(sum(r["title_recall"] for r in rows) / len(rows), 4),
                "grounded_ratio": round(sum(r["grounded_ratio"] for r in rows) / len(rows), 4),
            }
            for k, rows in sorted(
                {c["kind"]: [x for x in cases if x["kind"] == c["kind"]] for c in cases}.items()
            )
        },
    }


def report(rep: dict[str, Any]) -> None:
    """콘솔 출력. 제명·본문은 찍지 않는다."""
    typer.echo(f"[t5] {rep['source']}")
    typer.echo(
        f"     관보 {len(rep['issues'])}개 호 → 안건 {rep['items_total']}건"
        f" (색인 {rep['items_indexed']}건), 표본 {rep['sample']}건"
    )
    typer.echo(f"     종류 {rep['kinds']}")
    typer.echo(f"     임베딩 {rep['embed_model'] or '(미지정)'} / 요약 {rep['model'] or '(미지정)'}")
    typer.echo("")
    typer.echo("  검색 (제명으로 질의 → 그 안건이 상위에 오는가)")
    typer.echo(f"    {'방식':10} {'top-1':>7} {'top-3':>7} {'같은종류 top3':>13}")
    for name, m in rep["retrieval"].items():
        typer.echo(
            f"    {name:10} {m['self_at1']:6.1%} {m['self_at3']:6.1%} {m['same_kind_at3']:12.1%}"
            f"   ({m['elapsed_sec']}s)"
        )

    s = rep.get("summary", {})
    if "summary" in s:
        m = s["summary"]
        typer.echo("")
        typer.echo("  요약 (공식 제명·본문 근거로 대조)")
        typer.echo(f"    제명 재현율   {m['title_recall_mean']:.1%}")
        typer.echo(
            f"    숫자 근거율   {m['grounded_ratio_mean']:.1%}"
            f"  ({m['numbers_grounded']}/{m['numbers_total']} 토큰이 본문에 실재)"
        )
        typer.echo(f"    압축률        {m['compression_mean']:.1%}")
        if m["empty_outputs"]:
            typer.echo(f"    빈 출력       {m['empty_outputs']}건")
        typer.echo("    종류별:")
        for k, v in s["by_kind"].items():
            typer.echo(
                f"      {k:8} ({v['cases']:2}건) 제명 {v['title_recall']:6.1%}"
                f" | 근거 {v['grounded_ratio']:6.1%}"
            )
    else:
        typer.echo("")
        typer.echo(f"  요약: {s.get('skipped', '미실행')}")
    typer.echo("")
    typer.echo(f"  → {rep['results_path']}  ({rep['elapsed_sec']}s)")
