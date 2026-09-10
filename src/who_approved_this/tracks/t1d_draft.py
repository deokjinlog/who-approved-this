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
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
import typer

from who_approved_this.collect.opengokr_local import OpenGoKrLocalCollector
from who_approved_this.config import DATA_ROOT, track_data_dir, track_results_dir
from who_approved_this.evaluate.approval_match import load_gold, load_orgs
from who_approved_this.tracks.t1c_predictors import _THINK_BLOCK, ollama_chat
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

#: 초안 3개를 서로 다른 **스타일**로 뽑는다.
#: temperature 만 바꾸면 출력이 짧을 때 분포가 뾰족해 세 개가 바이트 단위로 같아졌다
#: (E2E D-2). 내용이 아니라 형식 축으로 나누는 게 "3개 중 고르기"에 맞다.
STYLES = (
    ("요약형", "핵심만 간결하게. 항목은 최소한으로.", 0.2),
    ("상세형", "항목을 나누어 자세히. 근거와 산출을 항목으로 편다.", 0.5),
    ("격식형", "공문 격식을 갖춰. 두괄식으로 쓰고 문장을 정중하게.", 0.4),
)
TEMPERATURES = tuple(t for _n, _d, t in STYLES)

#: 참고 본문을 프롬프트에 넣을 때의 길이 상한.
#: 길게 넣으면 모델이 대상 제목이 아니라 참고 문서 내용을 이어 쓴다(E2E D-3).
REF_CHARS = 800

#: 출력에 이게 남아 있으면 프롬프트가 샌 것으로 보고 실패로 기록한다.
PROMPT_MARKERS = ("## 참고문서", "# 작성할 문서", "본문:", "위 제목으로 문서 본문")

#: 내용 평가에 쓸 명사구 개수.
KEYPHRASE_N = 20

#: 초안 컨텍스트에서 사람 이름을 가릴 때 쓰는 대체 문자열.
NAME_MASK = "○○○"

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


DRAFT_SYSTEM = """너는 한국 행정기관의 공문서 작성자다. 주어진 참고 문서의 **형식**을 따라
새 문서의 본문 초안을 쓴다.

규칙:
- 참고 문서의 **내용을 옮기지 않는다.** 형식(항목 구성·번호 매김·말투)만 가져온다.
- 작성할 문서의 제목이 다루는 사안만 쓴다.
- 참고 문서에 없는 금액·날짜·수량·문서번호를 지어내지 않는다. 모르는 값은 (   ) 로 비운다.
- 사람 이름을 새로 만들지 않는다.
- 참고 문서의 머리글(기관명·수신)을 베끼지 않는다.

출력은 새 문서의 본문만. 설명·머리말·참고문서 표시를 붙이지 않는다."""


def has_prompt_leak(text: str) -> bool:
    """출력에 내부 프롬프트 마커가 남았는지. 남으면 실패로 본다."""
    return any(m in text for m in PROMPT_MARKERS)


def on_topic(text: str, title: str, head: int = 200) -> bool:
    """초안 앞부분이 대상 제목의 사안을 다루는지.

    제목의 내용어가 앞 ``head`` 자에 하나도 없으면 다른 문서 내용을 이어 쓴 것으로 본다.
    """
    words = [w for w in re.findall(r"[가-힣]{2,}", title) if len(w) >= 2]
    return not words or any(w in text[:head] for w in words)


def generate(
    model: str,
    prompt: str,
    temperature: float,
    timeout: int = 600,
    style: str = "",
) -> str:
    """ollama chat 으로 초안 하나를 만든다.

    ``/api/generate`` 가 아니라 ``/api/chat`` 을 쓴다 — generate 는 이어쓰기라
    프롬프트가 문서 모양이면 모델이 그 문서를 계속 이어 써서 내부 마커와
    참고 본문이 그대로 새어 나왔다(E2E D-1).
    """
    system = DRAFT_SYSTEM + (f"\n\n이번 초안의 스타일: {style}" if style else "")
    raw = ollama_chat(model, system, prompt, temperature=temperature, timeout=timeout)
    return _THINK_BLOCK.sub("", raw or "").strip()


SYSTEM = """너는 한국 행정기관의 공문서 작성자다. 제목과 같은 조직의 기존 문서를 참고해 문서 초안을 쓴다.

규칙:
- 같은 조직의 기존 문서 형식을 그대로 따른다. 항목 구성·번호 매김·말투를 참고 문서에서 가져온다.
- 참고 문서에 없는 금액·날짜·수량을 지어내지 않는다. 모르는 값은 (   ) 로 비워 둔다.
- 사람 이름을 새로 만들지 않는다.
- 제목이 다루는 사안만 쓴다.

출력은 문서 본문만. 설명이나 머리말을 붙이지 않는다."""


def mask_names(text: str, names: set[str]) -> str:
    """참고 본문의 사람 이름을 가린다.

    초안 단계에서 사람 이름을 미리 채우는 건 대체로 원하는 동작이 아니고,
    실측에서 24개 초안 중 14개에 참고 문서의 실명이 그대로 복사됐다.
    가릴 이름은 gold 에 적힌 명단을 쓴다(본문에서 이름을 추정하지 않는다).
    """
    out = text
    for name in sorted(names, key=len, reverse=True):
        if len(name) < 2:
            continue
        # "허채윤" 뿐 아니라 "허 채 윤", "허\n채\n윤" 처럼 글자 사이가 벌어진 표기도 잡는다.
        # 공문은 자간을 공백으로 벌려 찍는 일이 잦다(E2E D-4).
        spaced = r"\s*".join(re.escape(ch) for ch in name)
        out = re.sub(spaced, NAME_MASK, out)
    return out


def build_prompt(
    target: dict[str, Any], refs: list[dict[str, Any]], names: set[str] | None = None
) -> str:
    blocks = [
        f"[형식 참고 {i} · 유형 {c['doc_type']}]\n"
        + (mask_names(c["text"][:REF_CHARS], names) if names else c["text"][:REF_CHARS])
        for i, c in enumerate(refs, start=1)
    ]
    title = target["제목"]
    # 대상 제목을 **앞뒤로 두 번** 둔다. 참고 본문이 길면 모델이 마지막 문맥을
    # 이어 쓰는 경향이 있어, 마지막에 다시 못박아야 사안이 흔들리지 않는다(E2E D-3).
    return (
        f"작성할 문서 제목: {title}\n\n"
        f"아래는 같은 조직의 기존 문서 {len(refs)}건이다. **형식만** 참고한다.\n\n"
        + "\n\n".join(blocks)
        + f"\n\n---\n\n이제 위 형식을 따라 **다음 제목의 문서 본문**을 작성하라.\n"
        f"제목: {title}"
    )


def run(
    model: str | None = None,
    embed_model: str | None = None,
    n_drafts: int = len(TEMPERATURES),
) -> dict[str, Any]:
    """T1d를 돌린다. ``embed_model`` 이 없으면 bm25 만, ``model`` 이 없으면 검색까지만."""
    data_dir = track_data_dir("t1")
    out_dir = DATA_ROOT / TRACK
    results_dir = track_results_dir(TRACK)

    docs = load_docs(data_dir)
    gold = load_gold(data_dir / "approval_gold.tsv")
    for d in docs:
        d["cells"] = [c for items in gold.get(d["저장파일명"], {}).values() for c in items]
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

    gold_names = {
        c["name"]
        for d in docs
        for c in d.get("cells", [])
        if c.get("name") and c["name"] not in ("-", "?")
    }
    generation: dict[str, Any] = (
        {"skipped": "모델 미지정"}
        if not model
        else _generate_all(
            model, targets, chunks, retrievers[-1], out_dir, gold_names, n_drafts
        )
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
    names: set[str] | None = None,
    n_drafts: int = len(TEMPERATURES),
) -> dict[str, Any]:
    """대상마다 초안을 만들고 실제 문서와 대조한다.

    참고 문서는 **같은 조직으로 제한**한다. 검색 지표는 조직을 안 걸고 재지만
    (그게 검색 품질이다), 생성 컨텍스트에 다른 기관 문서가 들어가면 초안이
    그 기관 머리글을 그대로 베낀다 — 실측에서 초안 3개가 그렇게 오염됐다.
    """
    drafts_dir = out_dir / "drafts"
    review_dir = out_dir / "review"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    retries, leaks, off_topic = [0], [0], [0]
    t0 = time.perf_counter()
    for target in targets:
        refs = [
            c
            for c, _ in retriever.search(target["제목"], k=TOP_K * 6)
            if c["doc_id"] != target["doc_id"] and c["org"] == target["org"]
        ][:CONTEXT_K]
        prompt = build_prompt(target, refs, names)

        scored: list[tuple[dict[str, float], str, float]] = []
        for style_name, style_hint, temp in STYLES[:n_drafts]:
            draft = generate(model, prompt, temp, style=f"{style_name} — {style_hint}")
            # 프롬프트가 새거나 사안이 어긋나면 한 번만 다시 만든다.
            if has_prompt_leak(draft) or not on_topic(draft, target["제목"]):
                retries[0] += 1
                draft = generate(model, prompt, temp, style=f"{style_name} — {style_hint}")
            if has_prompt_leak(draft):
                leaks[0] += 1
            if not on_topic(draft, target["제목"]):
                off_topic[0] += 1
            (drafts_dir / f"{target['doc_id']}-{style_name}.md").write_text(
                draft, encoding="utf-8"
            )
            scored.append((score_draft(target["body"], draft), draft, temp))

        all_same = len({d for _sc, d, _t in scored}) == 1 and len(scored) > 1
        best = max(scored, key=lambda s: (s[0]["keyphrases_found"], s[0]["sections_found"]))
        (review_dir / f"{target['doc_id']}.md").write_text(
            f"# {target['doc_id']}\n\n## 실제 문서\n\n{target['body']}\n\n"
            f"---\n\n## 초안 (best, temperature={best[2]})\n\n{best[1]}\n",
            encoding="utf-8",
        )
        cases.append(
            {
                "doc_id": target["doc_id"],
                "best_temperature": best[2],
                "all_same": all_same,
                **best[0],
            }
        )

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
