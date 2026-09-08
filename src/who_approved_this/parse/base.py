"""파싱 어댑터의 공통 계약."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Parser(ABC):
    """PDF 한 건을 받아 텍스트 또는 구조화 결과를 돌려주는 어댑터.

    입출력 계약
    -----------
    입력
        ``pdf_path: Path`` — 원문 PDF 경로.
    출력
        * ``str`` — 텍스트 추출형 트랙(T4 OCR 등)의 전문(全文).
        * ``dict[str, Any]`` — 구조화 추출형 트랙(T1 결재선, T2 입찰 필드 등)의
          필드 dict. 키는 대응하는
          :class:`~who_approved_this.evaluate.base.Evaluator` 의 기대 키와 맞춘다.

        어느 쪽을 돌려주는지는 구현체마다 고정한다. 한 구현체가 호출마다
        타입을 바꾸지 않는다.

    규칙
    ----
    * 렌더링 해상도·모델명 등 재현에 필요한 설정은 생성자에서 받고
      :attr:`name` 에 드러낸다(예: ``"vision-ocr@200dpi"``).
    * 실패는 예외로 올린다. 빈 문자열로 삼키지 않는다.
    """

    #: 방식 식별자. 결과표의 "방식" 열에 그대로 쓰인다.
    name: str = "parser"

    @abstractmethod
    def parse(self, pdf_path: Path) -> str | dict[str, Any]:
        """``pdf_path`` 를 파싱해 전문 문자열 또는 필드 dict 를 돌려준다."""
        raise NotImplementedError

    def __call__(self, pdf_path: Path) -> str | dict[str, Any]:
        return self.parse(pdf_path)
