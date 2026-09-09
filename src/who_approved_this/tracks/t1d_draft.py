"""T1d — 품의서 초안 자동 생성 리허설.

회사 기능 ①의 리허설이다. **대상 문서의 제목만** 주고, 같은 조직의 다른 문서에서
검색해 초안을 쓰게 한 뒤 실제 문서와 비교한다(경기도서관 8건 leave-one-out).

두 단계를 따로 잰다.

1. **검색** — 제목으로 참고 문서를 제대로 찾는가.
   ``bm25`` / ``vector`` / ``hybrid`` 세 방식 비교.
   지표: 같은 조직 문서가 top-3 에 몇 건인지, 같은 유형이 top-3 에 있는지.
2. **생성** — 찾은 문서로 초안을 쓰면 실제 문서에 얼마나 가까운가.
   정답이 없으므로 **실제 문서를 기준**으로 구조·내용·길이를 잰다.

개인정보
    초안에는 참고 문서의 실명이 섞일 수 있어 **리포 밖**(``DATA_ROOT/t1d/``)에 둔다.
    ``results/t1d/`` JSON에는 숫자만 남긴다.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
import typer

from who_approved_this.collect.opengokr_local import OpenGoKrLocalCollector
from who_approved_this.config import DATA_ROOT, track_data_dir, track_results_dir
from who_approved_this.evaluate.approval_match import load_orgs
from who_approved_this.tracks.t1d_retrieval import (
    BM25Retriever,
    HybridRetriever,
    VectorRetriever,
    chunk_documents,
    doc_type,
)

TRACK = "t1d"

#: 초안을 만들 대상 조직. 문서가 8건이라 leave-one-out 이 성립하는 유일한 조직이다.
TARGET_ORG = "경기도서관"

#: 검색 결과 상위 몇 개를 프롬프트에 넣는지.
TOP_K = 5
CONTEXT_K = 3

#: 초안 3개를 서로 다른 temperature 로 뽑는다.
TEMPERATURES = (0.2, 0.6, 1.0)

#: 내용 평가에 쓸 명사구 개수.
KEYPHRASE_N = 20

#: 항목 머리글: "1. 목적", "2. 추진내용" 같은 번호 항목.
#: 날짜("2026. 6. 29.")가 같은 모양이라 오탐하기 쉬워, 머리글에 **한글 2자 이상**이
#: 있고 날짜로 시작하지 않을 때만 항목으로 본다.
_SECTION = re.compile(r"^[\s○·\-]*(\d{1,2})\s*[.)]\s*([^\n]{1,30})", re.M)
_DATEISH = re.compile(r"^\s*\d{2,4}\s*[.\-/]")

#: 내부결재 격자형 문서의 표 라벨. 항목 머리글이 아니다.
_TABLE_LABELS = ("등록번호", "등록일자", "결재일자", "공개구분", "접수", "시행")

#: 명사구 후보: 한글 2자 이상 덩어리.
_TOKEN = re.compile(r"[가-힣]{2,}")

#: 조사·형식어처럼 내용이 없는 낱말.
_STOP = {
    "합니다", "하고자", "다음과", "같이", "관련", "대하여", "따라", "위하여", "위한",
    "사항", "내용", "경우", "우리", "해당", "제출", "붙임", "기타", "아래", "통하여",
}


def load_docs(data_dir: Path) -> list[dict[str, Any]]:
    """20건 전체의 본문·조직·유형을 모은다."""
    orgs = load_orgs(data_dir / "approval_gold.tsv")
    docs: list[dict[str, Any]] = []
    for pdf_path, row in OpenGoKrLocalCollector(data_dir).items():
        with pymupdf.open(pdf_path) as doc:
            body = "\n".join(doc[i].get_text(sort=True) for i in range(doc.page_count))
        docs.append(
            {
                **row,
                "doc_id": pdf_path.stem,
                "org": orgs.get(row["저장파일명"], ""),
                "doc_type": doc_type(row["제목"]),
                "body": body,
            }
        )
    return docs


def ollama_embed(model: str) -> Any:
    """ollama 임베딩 함수를 만든다. 실패하면 ``None``."""
    import urllib.error
    import urllib.request

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            payload = json.dumps({"model": model, "prompt": text}).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                out.append(json.loads(resp.read())["embedding"])
        return out

    return embed


def _retrieval_cases(
    docs: list[dict[str, Any]], retriever: Any, targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """대상 제목으로 검색하고 top-3 품질을 잰다."""
    cases: list[dict[str, Any]] = []
    for target in targets:
        hits = [
            (c, s)
            for c, s in retriever.search(target["제목"], k=TOP_K * 3)
            if c["doc_id"] != target["doc_id"]
        ][:TOP_K]
        top3 = hits[:3]
        cases.append(
            {
                "doc_id": target["doc_id"],
                "same_org_in_top3": sum(1 for c, _ in top3 if c["org"] == target["org"]),
                "same_type_in_top3": int(
                    any(c["doc_type"] == target["doc_type"] for c, _ in top3)
                ),
                "distinct_docs_in_top5": len({c["doc_id"] for c, _ in hits}),
                "hits": [c["chunk_id"] for c, _ in hits],
            }
        )
    return cases


def _keyphrases(text: str, n: int = KEYPHRASE_N) -> list[str]:
    """실제 문서에서 자주 나오는 한글 낱말 n개. 내용 비교용."""
    counts = Counter(t for t in _TOKEN.findall(text) if t not in _STOP and len(t) >= 2)
    return [w for w, _ in counts.most_common(n)]


def _sections(text: str) -> list[str]:
    """번호 항목 머리글 목록("1. 목적" → "목적"). 날짜 줄은 뺀다."""
    out: list[str] = []
    for m in _SECTION.finditer(text):
        head = m.group(2).strip()
        hangul = len(re.findall(r"[가-힣]", head))
        if (
            hangul >= 2
            and not _DATEISH.match(head)
            and not any(label in head for label in _TABLE_LABELS)
        ):
            out.append(head)
    return out


def score_draft(actual: str, draft: str) -> dict[str, float]:
    """실제 문서를 기준으로 초안을 잰다. 정답이 없으므로 '닮은 정도'만 본다."""
    want = _sections(actual)
    got = " ".join(_sections(draft))
    hit_sections = sum(1 for s in want if s and s[:6] in got)
    phrases = _keyphrases(actual)
    hit_phrases = sum(1 for p in phrases if p in draft)
    return {
        "sections_expected": float(len(want)),
        "sections_found": float(hit_sections),
        "keyphrases_expected": float(len(phrases)),
        "keyphrases_found": float(hit_phrases),
        "length_ratio": round(len(draft) / max(len(actual), 1), 3),
    }


def generate(model: str, prompt: str, temperature: float, timeout: int = 300) -> str:
    """ollama 로 초안 하나를 만든다."""
    try:
        done = subprocess.run(
            ["ollama", "run", model, "--format", "", ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**__import__("os").environ, "OLLAMA_TEMPERATURE": str(temperature)},
        )
        return done.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


SYSTEM = """너는 한국 행정기관의 공문서 작성자다. 제목과 같은 조직의 기존 문서를 참고해 문서 초안을 쓴다.

규칙:
- 같은 조직의 기존 문서 형식을 그대로 따른다. 항목 구성·번호 매김·말투를 참고 문서에서 가져온다.
- 참고 문서에 없는 금액·날짜·수량을 지어내지 않는다. 모르는 값은 (   ) 로 비워 둔다.
- 사람 이름을 새로 만들지 않는다.
- 제목이 다루는 사안만 쓴다.

출력은 문서 본문만. 설명이나 머리말을 붙이지 않는다."""


def build_prompt(target: dict[str, Any], refs: list[dict[str, Any]]) -> str:
    blocks = [
        f"## 참고문서 {i}(유형: {c['doc_type']})\n본문:\n{c['text']}"
        for i, c in enumerate(refs, start=1)
    ]
    return (
        f"{SYSTEM}\n\n# 같은 조직의 기존 문서 {len(refs)}건\n\n"
        + "\n\n".join(blocks)
        + f"\n\n# 작성할 문서\n\n제목: {target['제목']}\n\n위 제목으로 문서 본문 초안을 작성하라."
    )


def run(model: str | None = None, embed_model: str | None = None) -> dict[str, Any]:
    """T1d를 돌린다. ``embed_model`` 이 없으면 bm25 만, ``model`` 이 없으면 검색까지만."""
    data_dir = track_data_dir("t1")
    out_dir = DATA_ROOT / TRACK
    results_dir = track_results_dir(TRACK)

    docs = load_docs(data_dir)
    chunks = chunk_documents(docs)
    targets = [d for d in docs if d["org"] == TARGET_ORG]

    bm25 = BM25Retriever(chunks)
    retrievers: list[Any] = [bm25]
    if embed_model:
        vector = VectorRetriever(chunks, ollama_embed(embed_model))
        retrievers += [vector, HybridRetriever(bm25, vector)]

    started = time.perf_counter()
    retrieval: dict[str, Any] = {}
    for retriever in retrievers:
        t0 = time.perf_counter()
        cases = _retrieval_cases(docs, retriever, targets)
        n = len(cases) or 1
        retrieval[retriever.name] = {
            "cases": [{k: v for k, v in c.items() if k != "hits"} for c in cases],
            "summary": {
                "cases": len(cases),
                "same_org_in_top3_mean": round(
                    sum(c["same_org_in_top3"] for c in cases) / n, 3
                ),
                "same_type_in_top3_rate": round(
                    sum(c["same_type_in_top3"] for c in cases) / n, 3
                ),
                "distinct_docs_in_top5_mean": round(
                    sum(c["distinct_docs_in_top5"] for c in cases) / n, 3
                ),
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            },
        }

    generation: dict[str, Any] = {"skipped": "모델 미지정"} if not model else _generate_all(
        model, targets, chunks, retrievers[-1], out_dir
    )

    now = datetime.now()
    report = {
        "track": TRACK,
        "run_at": now.isoformat(timespec="seconds"),
        "target_org": TARGET_ORG,
        "documents": len(docs),
        "chunks": len(chunks),
        "doc_types": dict(Counter(d["doc_type"] for d in docs)),
        "prompt_template": "src/who_approved_this/tracks/t1d_prompt.md",
        "embed_model": embed_model,
        "model": model,
        "retrieval": retrieval,
        "generation": generation,
        "note": "초안·참고 본문은 리포 밖 DATA_ROOT/t1d 에 둔다. 여기에는 숫자만.",
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


def _generate_all(
    model: str,
    targets: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    retriever: Any,
    out_dir: Path,
) -> dict[str, Any]:
    """대상마다 초안 3개를 만들고 실제 문서와 대조한다."""
    drafts_dir = out_dir / "drafts"
    review_dir = out_dir / "review"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for target in targets:
        refs = [
            c
            for c, _ in retriever.search(target["제목"], k=TOP_K * 3)
            if c["doc_id"] != target["doc_id"]
        ][:CONTEXT_K]
        prompt = build_prompt(target, refs)

        scored: list[tuple[dict[str, float], str, float]] = []
        for temp in TEMPERATURES:
            draft = generate(model, prompt, temp)
            (drafts_dir / f"{target['doc_id']}-t{temp}.md").write_text(
                draft, encoding="utf-8"
            )
            scored.append((score_draft(target["body"], draft), draft, temp))

        best = max(scored, key=lambda s: (s[0]["keyphrases_found"], s[0]["sections_found"]))
        (review_dir / f"{target['doc_id']}.md").write_text(
            f"# {target['doc_id']}\n\n## 실제 문서\n\n{target['body']}\n\n"
            f"---\n\n## 초안 (best, temperature={best[2]})\n\n{best[1]}\n",
            encoding="utf-8",
        )
        cases.append({"doc_id": target["doc_id"], "best_temperature": best[2], **best[0]})

    n = len(cases) or 1
    return {
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "section_recall": round(
                sum(c["sections_found"] for c in cases)
                / max(sum(c["sections_expected"] for c in cases), 1),
                3,
            ),
            "keyphrase_recall": round(
                sum(c["keyphrases_found"] for c in cases)
                / max(sum(c["keyphrases_expected"] for c in cases), 1),
                3,
            ),
            "length_ratio_mean": round(sum(c["length_ratio"] for c in cases) / n, 3),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
        "output_dir": str(out_dir),
    }


def report(rep: dict[str, Any]) -> None:
    """콘솔 출력. 값은 찍지 않는다."""
    typer.echo(f"[t1d] 대상 조직 {rep['target_org']} — 문서 {rep['documents']}건 → 청크 {rep['chunks']}개")
    typer.echo(f"      유형 분포 {rep['doc_types']}")
    typer.echo(f"      임베딩 {rep['embed_model'] or '(미지정)'} / 생성 {rep['model'] or '(미지정)'}")
    typer.echo("")
    typer.echo(f"  검색 (대상 제목으로 top-3, 케이스 {len(rep['retrieval'][next(iter(rep['retrieval']))]['cases'])})")
    typer.echo(f"    {'방식':10} {'같은조직 top3':>12} {'같은유형 top3':>12} {'top5 문서수':>11}")
    for name, data in rep["retrieval"].items():
        s = data["summary"]
        typer.echo(
            f"    {name:10} {s['same_org_in_top3_mean']:11.2f}/3 {s['same_type_in_top3_rate']:11.1%}"
            f" {s['distinct_docs_in_top5_mean']:10.2f}   ({s['elapsed_sec']}s)"
        )

    gen = rep.get("generation", {})
    if "summary" in gen:
        g = gen["summary"]
        typer.echo("")
        typer.echo("  생성 (실제 문서를 기준으로)")
        typer.echo(f"    항목 재현율   {g['section_recall']:.1%}")
        typer.echo(f"    핵심어 재현율 {g['keyphrase_recall']:.1%}")
        typer.echo(f"    길이 비율     {g['length_ratio_mean']:.2f}배")
        typer.echo(f"    → 사람 검토용: {gen['output_dir']}/review/")
    else:
        typer.echo("")
        typer.echo(f"  생성: {gen.get('skipped', '미실행')}")
    typer.echo("")
    typer.echo(f"  → {rep['results_path']}  ({rep['elapsed_sec']}s)")
