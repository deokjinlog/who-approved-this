"""수집 단계 — 공공 포털에서 원문 PDF와 공개 정답을 쌍으로 가져오는 어댑터들.

정답은 만들지 않는다. 원문(PDF)과 구조화 정답(메타데이터/텍스트)이 이미 공개된
소스만 고른다. 트랙이 늘어나면 이 패키지에 :class:`base.Collector` 구현체를 추가한다.
"""

from who_approved_this.collect.base import Collector

__all__ = ["Collector"]
