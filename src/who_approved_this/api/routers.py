"""HTTP 경로. 서비스를 호출해 스키마로 감싸기만 한다."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from who_approved_this.api import services
from who_approved_this.api.models import (
    AgentRequest,
    DocInfo,
    DraftRequest,
    DraftResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()


@router.get("/orgs", summary="문서가 있는 조직 목록")
def orgs() -> list[str]:
    return services.list_orgs()


@router.get("/docs", summary="문서 목록(실명 제외)")
def docs(org: str | None = None) -> list[DocInfo]:
    out = [services.get_doc(d["doc_id"]) for d in services.list_docs(org)]
    return [DocInfo(**d) for d in out if d]


@router.get("/docs/{doc_id}", summary="문서 메타(실명 제외)와 gold 결재선 유무")
def doc(doc_id: str) -> DocInfo:
    found = services.get_doc(doc_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"모르는 문서: {doc_id}")
    return DocInfo(**found)


@router.post("/approval-line/predict", summary="결재선 예측 (이름 마스킹 기본 켬)")
def predict(
    req: PredictRequest,
    exclude_doc_id: str | None = None,
    mask: bool = True,
) -> PredictResponse:
    """``mask`` 는 응답의 **사람 이름**만 가린다. 직위·조직은 그대로 둔다."""
    return PredictResponse(
        **services.predict_approval_line(
            req.org, req.title, req.body, exclude_doc_id, mask
        )
    )


@router.post("/draft/generate", summary="품의서 초안 생성")
def draft(req: DraftRequest, exclude_doc_id: str | None = None) -> DraftResponse:
    return DraftResponse(
        **services.generate_draft(req.org, req.title, req.n, exclude_doc_id)
    )


@router.post("/agent", summary="전자결재 에이전트 — 결재선·초안·요약·확인 필요 목록")
def agent(req: AgentRequest, mask: bool = True) -> dict:
    """사용자 식별 → 업무항목 → 결재선 → 템플릿 초안 → 요약 → 확인 필요.

    ``mask`` 는 응답의 사람 이름을 가린다(기본 켬).
    """
    return services.agent_handle(req.model_dump(exclude_none=True), mask)
