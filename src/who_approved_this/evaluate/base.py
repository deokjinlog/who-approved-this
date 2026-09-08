"""평가 어댑터의 공통 계약."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    """파싱 결과와 정답을 받아 점수 dict 를 돌려주는 대조기.

    입출력 계약
    -----------
    입력
        * ``parsed: str | dict[str, Any]`` — :class:`~who_approved_this.parse.base.Parser`
          가 돌려준 값 그대로.
        * ``ground_truth: dict[str, Any]`` — :class:`~who_approved_this.collect.base.Collector`
          가 돌려준 정답 dict 그대로.
    출력
        ``dict[str, float]`` — 지표 이름 → 값. 값은 JSON 직렬화 가능한 수치.
        예: ``{"cer": 0.041, "wer": 0.112}``, ``{"field_accuracy": 0.87}``.

    규칙
    ----
    * 지표 이름은 트랙 안에서 고정한다. 결과표 열 이름이 된다.
    * 점수 dict 에 원문 발췌나 실명을 넣지 않는다. 실패 케이스 샘플은
      결과 JSON의 별도 영역에 마스킹해 남긴다.
    """

    #: 지표 묶음 식별자. 결과표에 쓰인다.
    name: str = "evaluator"

    @abstractmethod
    def evaluate(
        self, parsed: str | dict[str, Any], ground_truth: dict[str, Any]
    ) -> dict[str, float]:
        """``parsed`` 를 ``ground_truth`` 와 대조해 지표 dict 를 돌려준다."""
        raise NotImplementedError

    def __call__(
        self, parsed: str | dict[str, Any], ground_truth: dict[str, Any]
    ) -> dict[str, float]:
        return self.evaluate(parsed, ground_truth)
