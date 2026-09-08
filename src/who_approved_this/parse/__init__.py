"""파싱 단계 — 렌더링·OCR·VLM 으로 PDF에서 텍스트나 구조를 뽑는 어댑터들.

로컬 처리를 우선한다. 상용 API는 비교군으로 소량만 붙인다.
"""

from who_approved_this.parse.base import Parser

__all__ = ["Parser"]
