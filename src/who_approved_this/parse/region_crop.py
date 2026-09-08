"""결재란 영역 찾기 + 잘라내기.

공공 결재문서의 결재란은 **두 자리 중 하나**에 있다. 10건 진단에서 확인된 두 양식:

* **시행문형**(7건) — 페이지 하단, 회색 띠 아래 "협조자 / 시행" 줄 위쪽에
  ``직위·이름`` 이 가로로 늘어선다.
* **내부결재 격자형**(2건) — 페이지 우측 상단에 표로 그려진다.
  (경기도서관-12039, 생물소재활용과-1332)

어느 쪽인지는 두 후보 영역에서 직위 문자열이 몇 개 잡히는지로 고른다.
문서별로 강제 지정하고 싶으면 :func:`approval_rect` 에 ``overrides`` 를 준다.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

#: 직위 판별용. 긴 것부터 봐야 "주사보"가 "주사"로 잘리지 않는다.
TITLE_HINTS = (
    "기록연구사", "환경연구사", "환경연구관", "본부장", "센터장", "담당관", "주사보",
    "연구관", "연구사", "전문위원", "서기관", "사무관", "주무관", "지사장", "교도소장",
    "부소장", "장관", "차관", "실장", "국장", "과장", "팀장", "관장", "원장", "소장",
    "부장", "계장", "차장", "주사", "서기", "교위", "교감", "청장", "시장", "군수",
)

#: 상단 후보 — 페이지 높이의 이 비율까지.
TOP_BAND = 0.30

#: 하단 후보 — 이 비율부터 이 비율까지. 주소·전화번호 줄은 뺀다.
BOTTOM_BAND = (0.70, 0.93)


def _count_titles(page: pymupdf.Page, rect: pymupdf.Rect) -> int:
    """영역 안에서 직위 문자열이 몇 개 잡히는지 센다."""
    text = page.get_text(clip=rect)
    return sum(1 for t in TITLE_HINTS if t in text)


def approval_rect(
    page: pymupdf.Page, overrides: dict[str, tuple[float, float]] | None = None,
    doc_id: str = "",
) -> tuple[pymupdf.Rect, str]:
    """결재란으로 보이는 영역과 그 종류(``"top"`` / ``"bottom"``)를 돌려준다.

    ``overrides`` 는 ``{doc_id: (y0_비율, y1_비율)}`` 로 문서별 강제 지정.
    """
    w, h = page.rect.width, page.rect.height
    if overrides and doc_id in overrides:
        y0, y1 = overrides[doc_id]
        return pymupdf.Rect(0, h * y0, w, h * y1), "override"

    top = pymupdf.Rect(0, 0, w, h * TOP_BAND)
    bottom = pymupdf.Rect(0, h * BOTTOM_BAND[0], w, h * BOTTOM_BAND[1])
    return (
        (top, "top") if _count_titles(page, top) > _count_titles(page, bottom)
        else (bottom, "bottom")
    )


def crop_approval(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    overrides: dict[str, tuple[float, float]] | None = None,
) -> tuple[Path, pymupdf.Rect, str]:
    """1페이지 결재란 영역을 ``dpi`` 로 렌더링해 PNG로 저장한다.

    돌려주는 값은 ``(png 경로, 잘라낸 영역, 영역 종류)``. 영역은 OCR 좌표를
    원본 페이지 좌표로 되돌릴 때 쓴다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{pdf_path.stem}-approval.png"
    with pymupdf.open(pdf_path) as doc:
        page = doc[0]
        rect, kind = approval_rect(page, overrides, pdf_path.stem)
        if not png.exists():
            page.get_pixmap(dpi=dpi, clip=rect).save(png)
    return png, rect, kind
