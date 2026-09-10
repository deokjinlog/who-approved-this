"""API 요청·응답 스키마.

개인정보
    ``/docs/{id}`` 는 **실명을 돌려주지 않는다**(직위와 유무만).
    예측·생성 결과에는 이름이 들어간다 — 로컬 데모 화면에서 눈으로 확인하기 위해서다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApprovalCell(BaseModel):
    """결재란 한 칸."""

    role: str = Field(description="기안 | 검토 | 결재 | 협조")
    title: str = Field(description="직위")
    name: str = Field(default="", description="이름. 모르면 빈 문자열")


class PredictRequest(BaseModel):
    org: str = Field(description="조직 키 (예: 경기도서관)")
    title: str
    body: str = Field(default="", description="본문. 규칙 기반 예측기는 쓰지 않는다")


class PredictResponse(BaseModel):
    masked: bool = Field(default=True, description="사람 이름을 가렸는지")
    org: str
    predictor: str
    cells: list[ApprovalCell]
    reference_doc_ids: list[str] = Field(description="예측에 참고한 문서 id")
    reference_count: int


class DraftRequest(BaseModel):
    org: str
    title: str
    n: int = Field(default=3, ge=1, le=5, description="초안 개수")


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    org: str
    doc_type: str
    score: float


class DraftResponse(BaseModel):
    org: str
    title: str
    retriever: str
    model: str | None
    drafts: list[str]
    references: list[RetrievedChunk]
    context_same_org_only: bool = True
    names_masked: bool = False
    note: str | None = None


class DocInfo(BaseModel):
    """문서 메타. **사람 이름은 담지 않는다**(직위·조직·수치만)."""

    doc_id: str
    org: str
    doc_type: str
    문서번호: str
    생산일자: str
    담당부서: str
    page_count: int
    body_chars: int
    has_gold_approval_line: bool
    gold_cell_count: int
    gold_titles: list[str] = Field(description="직위만. 이름은 제외한다")


class AgentRequest(BaseModel):
    """전자결재 에이전트 요청. 기안자는 이름 또는 (조직, 직위) 둘 중 하나로 준다."""

    user_name: str | None = Field(default=None, description="로그인 사용자 이름(명부 조회)")
    org: str | None = Field(default=None, description="이름 대신 조직을 직접 줄 때")
    title_of_user: str | None = Field(default=None, description="이름 대신 직위를 직접 줄 때")
    title: str = Field(description="작성할 문서 제목")
    body: str | None = Field(default=None, description="이미 쓴 본문이 있으면")
    attachments: list[str] | None = Field(default=None, description="첨부에서 뽑은 텍스트 목록")
    linked_docs: list[str] | None = Field(default=None, description="관련(선행) 문서번호")
    exclude_doc_id: str | None = Field(default=None, description="평가용 — 참고에서 뺄 문서")
