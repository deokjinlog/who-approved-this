"""T1 필드 대조 — 항목별 일치율.

T4의 CER과 재는 단위가 다르다. 결재문서는 글자를 얼마나 잘 읽었느냐가 아니라
**필드를 제대로 뽑았느냐**가 문제이므로, 지표는 항목별 일치(bool)와 그 비율이다.

개인정보
--------
비교 대상 값(제목·담당자 실명 등)은 이 모듈 안에서만 쓰고 **밖으로 내보내지 않는다**.
돌려주는 건 일치 여부와 개수뿐이다. 실패 원인도 값이 아니라 사유 코드로만 남긴다.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from who_approved_this.evaluate.base import Evaluator

#: T1a 채점 대상. metadata.tsv 헤더 그대로.
FIELDS = ("제목", "생산기관", "담당부서", "담당자", "생산일자", "문서번호")

#: 결재선에서 세는 역할.
ROLES = ("기안", "검토", "결재", "협조")

_PUNCT = re.compile(r"[()\[\]{}<>「」『』.,·:;/\\'\"’”“‘\-–—]")
_WS = re.compile(r"\s+")


def normalize_field(value: str) -> str:
    """공백·괄호·마침표를 걷어낸 비교용 표현.

    한국어 공문서는 같은 값을 "총무과", "총 무 과", "(총무과)" 처럼 다르게 적는다.
    이 차이로 불일치가 나면 추출 성능이 아니라 표기 관습을 재게 된다.
    """
    out = unicodedata.normalize("NFC", str(value))
    out = _PUNCT.sub("", out)
    return _WS.sub("", out).strip()


def _reason(expected: str, actual: str) -> str:
    """실패 사유를 **값 없이** 코드로만 돌려준다."""
    if not actual:
        return "미추출"
    if not expected:
        return "정답없음"
    if normalize_field(actual) in normalize_field(expected):
        return "부분일치(추출값이 정답의 일부)"
    if normalize_field(expected) in normalize_field(actual):
        return "부분일치(추출값이 정답보다 김)"
    return "불일치"


class FieldMatchEvaluator(Evaluator):
    """추출 필드를 포털 메타데이터와 대조해 항목별 일치율을 낸다.

    입력
        * ``parsed: dict`` — :class:`~who_approved_this.parse.fields_textlayer.FieldTextLayerParser` 출력.
        * ``ground_truth: dict`` — metadata.tsv 한 행.
    출력
        ``{"<필드>_exact": 0/1, "<필드>_norm": 0/1, "field_accuracy": 0~1,
        "approval_line_count": n, "approval_<역할>_count": n}``.

        값은 하나도 담지 않는다. 실패 사유는 :attr:`failures` 에 따로 모아
        트랙이 콘솔로만 보여준다.
    """

    name = "field-match"

    def __init__(self) -> None:
        #: 마지막 evaluate 의 실패 목록. ``[(필드, 사유)]`` — 값은 없다.
        self.failures: list[tuple[str, str]] = []

    def evaluate(
        self, parsed: str | dict[str, Any], ground_truth: dict[str, Any]
    ) -> dict[str, float]:
        """필드별 일치 여부와 결재선 통계를 돌려준다."""
        if not isinstance(parsed, dict):
            raise TypeError(f"{type(self).__name__} 은 dict 결과만 받는다: {type(parsed)}")

        self.failures = []
        scores: dict[str, float] = {}
        hits = 0
        for field in FIELDS:
            expected = str(ground_truth.get(field, ""))
            actual = str(parsed.get(field, ""))
            exact = expected == actual and bool(expected)
            norm = bool(expected) and normalize_field(expected) == normalize_field(actual)
            scores[f"{field}_exact"] = float(exact)
            scores[f"{field}_norm"] = float(norm)
            hits += int(norm)
            if not norm:
                self.failures.append((field, _reason(expected, actual)))
        scores["field_accuracy"] = hits / len(FIELDS)

        line = parsed.get("approval_line") or []
        scores["approval_line_count"] = float(len(line))
        for role in ROLES:
            scores[f"approval_{role}_count"] = float(
                sum(1 for c in line if c.get("role") == role)
            )
        return scores
