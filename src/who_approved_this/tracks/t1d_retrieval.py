"""T1d 검색 단계 — 청킹과 세 가지 검색 방식.

회사 기능 ①(품의서 초안 자동 생성)의 앞단이다. 초안을 쓰려면 먼저
**무엇을 참고할지 찾아야** 하고, 그 검색이 제대로 되는지를 따로 잰다.

세 방식을 같은 인터페이스로 비교한다.

    ``bm25``    단어 빈도 기반. 모델이 필요 없다.
    ``vector``  로컬 임베딩 코사인 유사도.
    ``hybrid``  두 순위를 RRF(Reciprocal Rank Fusion)로 합친다.

한국어 토큰화
    형태소 분석기를 새로 붙이지 않는다(설치 부담이 크고 골격 밖이다). 대신
    **띄어쓰기 토큰 + 글자 바이그램**을 함께 쓴다. 조사가 붙어 "도서관을"과
    "도서관이"가 다른 토큰이 되는 문제를 바이그램이 메운다.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from rank_bm25 import BM25Okapi

#: 청크 최소·최대 길이(글자). 너무 짧은 조각은 검색 노이즈가 된다.
MIN_CHUNK = 40
MAX_CHUNK = 600

#: RRF 상수. 순위 합칠 때 상위권 가중을 조절한다(관례값 60).
RRF_K = 60

_WS = re.compile(r"\s+")


def doc_type(title: str) -> str:
    """제목으로 문서 유형을 가른다. 규칙은 단순하게, 순서대로 처음 걸리는 것.

    검색이 "같은 성격의 문서"를 찾아오는지 보려면 유형 꼬리표가 필요하다.
    """
    t = _WS.sub("", title)
    if "지출" in t:
        return "지출"
    if "계획" in t:
        return "계획"
    if "결과보고" in t or "회의록" in t:
        return "결과보고"
    if any(k in t for k in ("알림", "통보", "제출", "요청", "승인")):
        return "알림"
    return "기타"


def tokenize(text: str) -> list[str]:
    """띄어쓰기 토큰 + 글자 바이그램.

    한국어는 조사가 붙어 같은 낱말이 다른 토큰이 된다("도서관을"/"도서관이").
    바이그램을 섞으면 형태소 분석기 없이도 그 차이를 상당 부분 흡수한다.
    """
    words = [w for w in _WS.sub(" ", text).strip().split(" ") if w]
    flat = "".join(words)
    bigrams = [flat[i : i + 2] for i in range(len(flat) - 1)]
    return words + bigrams


def chunk_documents(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """문서 본문을 문단 단위로 쪼개고 문서 id·조직·유형 꼬리표를 붙인다."""
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc["body"]) if p.strip()]
        buffer = ""
        for para in paragraphs + [""]:
            if buffer and (not para or len(buffer) + len(para) > MAX_CHUNK):
                if len(buffer) >= MIN_CHUNK:
                    chunks.append(_chunk(doc, buffer, len(chunks)))
                buffer = ""
            buffer = f"{buffer}\n{para}".strip() if buffer else para
        if len(buffer) >= MIN_CHUNK:
            chunks.append(_chunk(doc, buffer, len(chunks)))
    return chunks


def _chunk(doc: dict[str, Any], text: str, idx: int) -> dict[str, Any]:
    return {
        "chunk_id": f"{doc['doc_id']}#{idx}",
        "doc_id": doc["doc_id"],
        "org": doc["org"],
        "doc_type": doc["doc_type"],
        "text": text,
    }


class BM25Retriever:
    """단어 빈도 기반 검색. 모델이 필요 없어 항상 돌릴 수 있는 기준선이다."""

    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class VectorRetriever:
    """로컬 임베딩 코사인 유사도 검색.

    ``embed`` 는 문자열 목록을 받아 벡터 목록을 돌려주는 함수다. 어떤 모델을 쓰든
    이 함수만 맞추면 된다(ollama / sentence-transformers 무관).
    """

    name = "vector"

    def __init__(
        self, chunks: list[dict[str, Any]], embed: Callable[[list[str]], list[list[float]]]
    ) -> None:
        self.chunks = chunks
        self.embed = embed
        self.vectors = _normalize(embed([c["text"] for c in chunks]))

    def search(self, query: str, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        q = _normalize(self.embed([query]))[0]
        scores = [sum(a * b for a, b in zip(q, v)) for v in self.vectors]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class HybridRetriever:
    """BM25 와 벡터 순위를 RRF 로 합친다.

    점수 스케일이 서로 달라 그대로 더할 수 없으므로 **순위**만 쓴다.
    """

    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever) -> None:
        self.bm25 = bm25
        self.vector = vector

    def search(self, query: str, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        fused: dict[str, float] = {}
        holder: dict[str, dict[str, Any]] = {}
        for retriever in (self.bm25, self.vector):
            for rank, (chunk, _score) in enumerate(retriever.search(query, k=k * 4)):
                fused[chunk["chunk_id"]] = fused.get(chunk["chunk_id"], 0.0) + 1.0 / (
                    RRF_K + rank + 1
                )
                holder[chunk["chunk_id"]] = chunk
        best = sorted(fused, key=lambda c: -fused[c])[:k]
        return [(holder[c], fused[c]) for c in best]


def _normalize(vectors: list[list[float]]) -> list[list[float]]:
    """코사인 유사도를 내적으로 계산하려고 미리 단위벡터로 만든다."""
    out = []
    for v in vectors:
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out
