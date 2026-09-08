"""로컬 OCR — Apple Vision (``ocrmac`` 래퍼).

모델 다운로드 없이 macOS 내장 엔진을 쓰므로 M2에서 바로 돌고, 원문이 로컬 밖으로
나가지 않는다(CLAUDE.md: 로컬 처리 우선). macOS 종속이라 리눅스로 옮길 때는
:class:`~who_approved_this.parse.base.Parser` 를 구현한 다른 파서를 ``parse/`` 에
추가한다 — 골격은 건드리지 않는다.

여기서는 OCR 엔진을 **하나만** 붙인다. 비교군(상용 API 등)은 별도 파서로,
별도 실행에서 붙인다.
"""

from __future__ import annotations

from pathlib import Path

from ocrmac import ocrmac

from who_approved_this.parse.base import Parser
from who_approved_this.parse.render import DEFAULT_DPI, render_pdf

#: 한국어 우선, 영문 혼용 문서를 위해 en-US 를 뒤에 둔다.
DEFAULT_LANGUAGES = ["ko-KR", "en-US"]


def ocr_image(png_path: Path, languages: list[str]) -> str:
    """PNG 한 장을 Apple Vision으로 읽어 줄 단위로 이어붙인 텍스트를 돌려준다."""
    result = ocrmac.OCR(
        str(png_path),
        recognition_level="accurate",
        language_preference=list(languages),
    ).recognize()

    # ocrmac 은 (text, confidence, [x, y, w, h]) 를 준다. 좌표는 좌하단 원점의
    # 정규화 좌표라, 읽는 순서(위→아래, 왼→오른쪽)로 다시 세운다.
    lines = sorted(result, key=lambda r: (-round(r[2][1], 2), r[2][0]))
    return "\n".join(text for text, _conf, _box in lines)


class AppleVisionOCRParser(Parser):
    """PDF를 렌더링한 뒤 Apple Vision OCR로 전문(全文)을 뽑는 파서.

    입력 ``pdf_path: Path`` → 출력 ``str`` (페이지 사이는 빈 줄로 구분).
    페이지별 점수가 필요한 트랙은 :meth:`parse_pages` 로 페이지 리스트를 받는다.
    """

    def __init__(
        self,
        render_root: Path,
        dpi: int = DEFAULT_DPI,
        languages: list[str] | None = None,
    ) -> None:
        self.render_root = render_root
        self.dpi = dpi
        self.languages = languages or list(DEFAULT_LANGUAGES)
        self.name = f"apple-vision(ocrmac)@{dpi}dpi"

    def parse(self, pdf_path: Path, max_pages: int | None = None) -> str:
        """``pdf_path`` 를 렌더링·OCR한 전문을 돌려준다."""
        return "\n\n".join(self.parse_pages(pdf_path, max_pages=max_pages))

    def parse_pages(self, pdf_path: Path, max_pages: int | None = None) -> list[str]:
        """페이지별 OCR 텍스트를 순서대로 돌려준다."""
        pages = render_pdf(pdf_path, self.render_root, dpi=self.dpi, max_pages=max_pages)
        return [ocr_image(p, self.languages) for p in pages]
