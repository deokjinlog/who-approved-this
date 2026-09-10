"""API 서비스 계층 — 트랙 코드를 불러 결과를 만든다.

여기서 로직을 새로 쓰지 않는다. 예측은 :mod:`~who_approved_this.tracks.t1c_predictors`,
검색·생성은 :mod:`~who_approved_this.tracks.t1d_draft` 를 그대로 호출한다.
문서·청크는 프로세스 수명 동안 한 번만 읽어 재사용한다(데모라 갱신 감시는 없다).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import pymupdf

from who_approved_this.config import track_data_dir
from who_approved_this.tracks import t1c_approval_predict as t1c
from who_approved_this.tracks import t1d_draft as t1d
from who_approved_this.tracks.t1c_predictors import BaselineVote, VoteTitlesLLMNames
from who_approved_this.tracks.t1d_retrieval import (
    BM25Retriever,
    HybridRetriever,
    VectorRetriever,
    chunk_documents,
)

#: 환경변수로 모델을 지정한다. 없으면 규칙 기반·BM25 로만 동작한다.
LLM_MODEL = os.environ.get("WAT_LLM_MODEL") or None
EMBED_MODEL = os.environ.get("WAT_EMBED_MODEL") or None


def warm_up() -> dict[str, Any]:
    """서버 기동 시 한 번 호출해 첫 요청 지연을 없앤다.

    문서·색인을 미리 만들고 모델을 한 번 깨운다. E2E 에서 첫 결재선 예측이
    18.9초, 이후 4.5초였다 — 그 차이가 여기서 사라진다.
    """
    import time as _time

    started = _time.perf_counter()
    _docs()
    _retriever()
    if EMBED_MODEL:
        t1d.ollama_embed(EMBED_MODEL)(["웜업"])
    if LLM_MODEL:
        # 짧은 "ping" 으로는 부족하다. 첫 요청이 느린 건 모델 로딩이 아니라
        # **긴 프롬프트 처리**(참고 7건 × 본문)라, 실제와 같은 크기로 한 번 돌려야
        # 프롬프트 캐시가 채워진다. 문서가 가장 많은 조직으로 웜업한다.
        counts: dict[str, int] = {}
        for d in _docs():
            counts[d["org"]] = counts.get(d["org"], 0) + 1
        if counts:
            org = max(counts, key=lambda k: counts[k])
            predict_approval_line(org, "웜업 요청", exclude_doc_id=None)
    return {"elapsed_sec": round(_time.perf_counter() - started, 2)}


@lru_cache(maxsize=1)
def _docs() -> list[dict[str, Any]]:
    """20건 문서(본문·조직·유형·gold 결재선)를 한 번만 읽는다."""
    data_dir = track_data_dir("t1")
    docs = t1d.load_docs(data_dir)  # 본문·조직·유형
    cells = {d["doc_id"]: d["cells"] for d in t1c.load_docs(data_dir)}  # gold 결재선
    for d in docs:
        d["cells"] = cells.get(d["doc_id"], [])
    return docs


@lru_cache(maxsize=1)
def _gold_names() -> frozenset[str]:
    """초안 컨텍스트에서 가릴 사람 이름(gold 명단)."""
    return frozenset(
        c["name"]
        for d in _docs()
        for c in d.get("cells", [])
        if c.get("name") and c["name"] not in ("-", "?")
    )


@lru_cache(maxsize=1)
def _retriever() -> Any:
    """임베딩 모델이 있으면 하이브리드, 없으면 BM25."""
    chunks = chunk_documents(_docs())
    bm25 = BM25Retriever(chunks)
    if not EMBED_MODEL:
        return bm25
    vector = VectorRetriever(chunks, t1d.ollama_embed(EMBED_MODEL))
    return HybridRetriever(bm25, vector)


def list_orgs() -> list[str]:
    """문서가 있는 조직 목록."""
    return sorted({d["org"] for d in _docs() if d["org"]})


def list_docs(org: str | None = None) -> list[dict[str, Any]]:
    """문서 목록. ``org`` 를 주면 그 조직만."""
    return [d for d in _docs() if not org or d["org"] == org]


def get_doc(doc_id: str) -> dict[str, Any] | None:
    """문서 메타. **실명은 빼고** 직위만 돌려준다."""
    doc = next((d for d in _docs() if d["doc_id"] == doc_id), None)
    if doc is None:
        return None
    with pymupdf.open(doc["pdf"] if "pdf" in doc else track_data_dir("t1") / doc["저장파일명"]) as f:
        pages = f.page_count
    return {
        "doc_id": doc["doc_id"],
        "org": doc["org"],
        "doc_type": doc["doc_type"],
        "문서번호": doc.get("문서번호", ""),
        "생산일자": doc.get("생산일자", ""),
        "담당부서": doc.get("담당부서", ""),
        "page_count": pages,
        "body_chars": len(doc["body"]),
        "has_gold_approval_line": bool(doc["cells"]),
        "gold_cell_count": len(doc["cells"]),
        "gold_titles": [c["title"] for c in doc["cells"]],
    }


def predict_approval_line(
    org: str,
    title: str,
    body: str = "",
    exclude_doc_id: str | None = None,
    mask: bool = True,
) -> dict[str, Any]:
    """같은 조직의 과거 문서를 참고해 결재선을 예측한다.

    모델이 지정돼 있으면 LLM, 아니면 다수결 baseline 을 쓴다.
    ``exclude_doc_id`` 는 leave-one-out 데모용 — 자기 자신을 참고에서 뺀다.

    ``mask`` 는 응답의 **사람 이름**만 가린다(기본 켬). 직위는 가리지 않는다.
    API 는 화면 밖으로 나가는 경로라 기본을 켜 둔다. 로컬 데모에서 gold 와
    눈으로 비교할 때만 끈다.
    """
    refs = [
        d
        for d in _docs()
        if d["org"] == org and d["cells"] and d["doc_id"] != exclude_doc_id
    ]
    # 기본 예측기는 결합(규칙으로 뼈대, LLM 으로 이름). 12케이스 + 재표집 40회에서
    # 직위·이름·칸수 모두 단독 방식보다 낫거나 같았다.
    predictor = VoteTitlesLLMNames(LLM_MODEL) if LLM_MODEL else BaselineVote()
    target = {"제목": title, "body": body, "org": org}
    cells = predictor.predict(refs, target) if refs else []
    if mask:
        cells = [
            {**c, "name": t1d.NAME_MASK if c.get("name") else ""} for c in cells
        ]
    return {
        "masked": mask,
        "org": org,
        "predictor": predictor.name,
        "cells": cells,
        "reference_doc_ids": [d["doc_id"] for d in refs],
        "reference_count": len(refs),
    }


def generate_draft(
    org: str, title: str, n: int = 3, exclude_doc_id: str | None = None
) -> dict[str, Any]:
    """제목으로 참고 문서를 검색하고 초안을 만든다.

    생성 모델이 없으면 초안 없이 **검색 결과만** 돌려준다(검색 단계는 모델 없이 돈다).
    """
    retriever = _retriever()
    hits = [
        (c, s)
        for c, s in retriever.search(title, k=t1d.TOP_K * 6)
        if c["doc_id"] != exclude_doc_id
    ][: t1d.TOP_K]
    # 생성 컨텍스트는 같은 조직으로 제한한다(다른 기관 머리글이 초안에 복사되는 걸 막는다).
    refs = [c for c, _ in hits if c["org"] == org][: t1d.CONTEXT_K]

    drafts: list[str] = []
    note = None
    if LLM_MODEL and refs:
        # 참고 본문의 사람 이름은 가린다(초안에 실명이 복사되지 않게).
        prompt = t1d.build_prompt({"제목": title}, refs, set(_gold_names()))
        for temp in t1d.TEMPERATURES[:n]:
            drafts.append(t1d.generate(LLM_MODEL, prompt, temp))
    elif not LLM_MODEL:
        note = "생성 모델 미지정(WAT_LLM_MODEL). 검색 결과만 돌려준다."

    return {
        "org": org,
        "title": title,
        "retriever": retriever.name,
        "model": LLM_MODEL,
        "drafts": drafts,
        "context_same_org_only": True,
        "names_masked": bool(LLM_MODEL),
        "references": [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "org": c["org"],
                "doc_type": c["doc_type"],
                "score": round(float(s), 4),
            }
            for c, s in hits
        ],
        "note": note,
    }


def gold_cells(doc_id: str) -> list[dict[str, str]]:
    """gold 결재선(이름 포함). 로컬 데모 화면 비교용."""
    doc = next((d for d in _docs() if d["doc_id"] == doc_id), None)
    return doc["cells"] if doc else []


def actual_body(doc_id: str) -> str:
    """실제 문서 본문. 로컬 데모 화면 비교용."""
    doc = next((d for d in _docs() if d["doc_id"] == doc_id), None)
    return doc["body"] if doc else ""
