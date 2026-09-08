"""who-approved-this — 공개 공공데이터로 문서 파싱 정확도를 재는 벤치마크 파이프라인.

공통 구조는 3단계로 고정한다::

    collect  →  parse  →  evaluate
    (원문 PDF + 정답)  (텍스트/구조 추출)  (점수 dict)

새 트랙은 각 단계의 어댑터(:class:`~who_approved_this.collect.base.Collector`,
:class:`~who_approved_this.parse.base.Parser`,
:class:`~who_approved_this.evaluate.base.Evaluator` 구현체)만 추가하고,
골격 자체는 바꾸지 않는다.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
