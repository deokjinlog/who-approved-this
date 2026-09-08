"""수집 어댑터의 공통 계약."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class Collector(ABC):
    """원문 PDF 경로와 공개 정답을 쌍으로 내놓는 소스 어댑터.

    입출력 계약
    -----------
    입력
        생성자에서 받는 소스별 설정(데이터 디렉터리, 조회 조건, 수집 개수 등).
        수집한 파일은 리포 밖 :data:`~who_approved_this.config.DATA_ROOT` 아래에 둔다.
    출력
        :meth:`items` 가 ``(pdf_path, ground_truth)`` 튜플을 하나씩 yield 한다.

        * ``pdf_path`` — 로컬에 존재하는 원문 PDF 경로.
        * ``ground_truth`` — 공개 정답 dict. 키는 트랙이 정하고,
          평가 단계의 :class:`~who_approved_this.evaluate.base.Evaluator` 와
          맞춘다. 예: T4 는 ``{"text": "<정답 전문>"}``,
          T1 은 ``{"title": ..., "dept": ..., "approval_line": [...]}``.

    규칙
    ----
    * ``ground_truth`` 에 기안자 실명 등 개인정보를 담지 않는다. 필요하면
      수집 단계에서 마스킹한다.
    * 포털 이용약관과 요청 간격을 지킨다. 대량 수집보다 양식별 대표 샘플.
    """

    #: 트랙/소스 식별자. 결과 파일명과 결과표에 쓰인다.
    name: str = "collector"

    @abstractmethod
    def items(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        """``(pdf_path, ground_truth)`` 쌍을 순서대로 yield 한다."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        return self.items()
