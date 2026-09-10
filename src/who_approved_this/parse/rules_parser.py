"""위임전결규정·조직도 문서 → ``rules/approval.yaml`` 변환기 (골격).

**아직 구현하지 않았다.** 근거 문서(위임전결규정 별표, 조직도, 직원 안내)가
도착하면 이 시그니처대로 채운다. 지금은 무엇을 어떤 모양으로 뽑을지만 고정해 둔다.

왜 규정 파싱이 필요한가
    지금까지의 규칙은 전부 **gold 이력에서 귀납**한 것이다("경기도서관은 관장 전결").
    이력 귀납은 관측된 문서 안에서만 맞고, 금액 구간처럼 표본에 안 나온 조건은
    아예 알 수 없다. 규정 문서에는 그 조건이 표로 들어 있다.
    두 벌을 같은 스키마로 만들어 차이를 재는 게 목적이다
    (:mod:`~who_approved_this.evaluate.rules_diff`).

입력 형태
    * 위임전결규정 **별표** — hwp/hwpx 또는 pdf. "사무 구분 × 전결권자" 표.
    * 조직도·직원 안내 — 조직 → 부서 → 직위 → 이름.
    * 기록물 분류기준표 — 문서 유형 분류에 쓸 수 있다(선택).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_document(path: Path) -> dict[str, Any]:
    """규정 문서 하나를 열어 표와 본문 텍스트로 만든다.

    확장자에 따라 갈린다 — hwpx 는 zip 안 XML, hwp 는 LibreOffice 변환 후 pdf 경로,
    pdf 는 기존 :mod:`~who_approved_this.parse.router` 를 탄다.

    돌려주는 값::

        {"path", "kind": "hwp|hwpx|pdf", "text": str,
         "tables": [[[cell, ...], ...], ...], "status": "ok|변환 불가"}

    변환이 안 되면 예외를 내지 않고 ``status`` 에 남긴다. 한 문서가 막혀도
    나머지 규정 파싱이 멈추면 안 된다.
    """
    raise NotImplementedError("근거 문서 도착 후 구현")


def extract_delegation_rules(tables: list[Any], org_key: str) -> list[dict[str, Any]]:
    """전결 별표에서 ``delegation_rule`` 목록을 뽑는다.

    별표는 보통 "사무의 종류 / 기관장 / 부기관장 / 과장" 열에 ○ 표시가 찍힌
    행렬 형태다. 행이 문서 유형, 열이 전결권자 직위이므로 ○ 위치가 곧 규칙이다.
    금액 구간은 행 제목에 "5천만원 이상" 처럼 붙어 있어 별도로 떼어낸다.
    """
    raise NotImplementedError("근거 문서 도착 후 구현")


def extract_chain(tables: list[Any], text: str, org_key: str) -> dict[str, Any]:
    """조직도·직원 안내에서 결재 계층(기안→검토→결재 직위)을 만든다."""
    raise NotImplementedError("근거 문서 도착 후 구현")


def extract_cooperation_rules(tables: list[Any], text: str) -> list[dict[str, Any]]:
    """규정 본문에서 협조 요건을 뽑는다("지출은 회계부서 협조" 등)."""
    raise NotImplementedError("근거 문서 도착 후 구현")


def extract_roster(tables: list[Any], text: str) -> list[dict[str, str]]:
    """직원 안내에서 ``(조직, 직위, 이름)`` 명부를 만든다.

    **실명이 나오므로 리포에 저장하지 않는다.** 호출부가
    ``DATA_ROOT/t1/roster.tsv`` 로 쓴다.
    """
    raise NotImplementedError("근거 문서 도착 후 구현")


def build_official_rules(paths: list[Path], org_key: str) -> dict[str, Any]:
    """규정 문서들을 읽어 ``rules/approval.yaml`` 스키마 dict 를 만든다.

    ``meta.source`` 는 ``"official"``, ``meta.source_docs`` 에 근거 파일명과
    쪽수를 남긴다. 규칙마다 ``note`` 에 조문·별표 번호를 붙여
    나중에 "이 칸이 왜 이렇게 정해졌나"를 사람이 되짚을 수 있게 한다.
    """
    raise NotImplementedError("근거 문서 도착 후 구현")


def build_induced_rules(gold_path: Path) -> dict[str, Any]:
    """gold 이력에서 같은 스키마를 **귀납**해 만든다(비교군).

    ``meta.source`` 는 ``"induced"``. 규정 문서 없이 관측만으로 어디까지 복원되는지가
    이 함수의 결과이고, 규정본과의 차이가 곧 "규정 없이 가면 틀리는 지점"이다.
    이건 지금 데이터로도 만들 수 있어 먼저 구현한다.
    """
    raise NotImplementedError("gold 기반 귀납 — ⑤ 단계에서 구현")
