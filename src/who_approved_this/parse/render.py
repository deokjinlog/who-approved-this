"""PDF → PNG 렌더링 (pymupdf).

OCR·VLM 파서가 공통으로 쓰는 전처리. 해상도 기본값은 200dpi로,
T4(관보 OCR) 기준선이자 다른 트랙의 출발점이다.

출력은 ``<out_root>/<pdf stem>/p{n}.png`` (1부터). 이미 있으면 다시 그리지 않고
재사용한다 — OCR을 여러 번 돌릴 때 렌더링이 매번 반복되지 않게.
같은 디렉터리의 ``_render.json`` 에 dpi를 적어 두고, dpi가 바뀌면 이전 PNG를
재사용하지 않고 새로 그린다(해상도가 섞인 채로 점수가 나오는 걸 막는다).
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

#: 트랙 공통 기본 렌더링 해상도.
DEFAULT_DPI = 200

#: 렌더링 조건을 적어 두는 사이드카 파일 이름.
_MARKER = "_render.json"


def render_pdf(
    pdf_path: Path,
    out_root: Path,
    dpi: int = DEFAULT_DPI,
    page_numbers: list[int] | None = None,
) -> list[Path]:
    """``pdf_path`` 의 지정 페이지를 ``dpi`` 로 렌더링하고 PNG 경로 목록을 돌려준다.

    ``out_root`` 는 리포 밖 :data:`~who_approved_this.config.DATA_ROOT` 아래를 쓴다
    (중간 산출물은 커밋하지 않는다). ``page_numbers`` 는 **1부터 세는 페이지 번호**
    목록이고, 생략하면 전체를 그린다. 파일명이 실제 페이지 번호를 쓰므로
    띄엄띄엄 고른 페이지도 캐시가 그대로 맞는다.
    """
    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    reusable = _marker_matches(out_dir, dpi)
    pngs: list[Path] = []
    with pymupdf.open(pdf_path) as doc:
        wanted = page_numbers if page_numbers is not None else range(1, doc.page_count + 1)
        for n in wanted:
            png = out_dir / f"p{n}.png"
            if not (reusable and png.exists()):
                doc[n - 1].get_pixmap(dpi=dpi).save(png)
            pngs.append(png)

    (out_dir / _MARKER).write_text(
        json.dumps({"dpi": dpi, "source": pdf_path.name}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return pngs


def _marker_matches(out_dir: Path, dpi: int) -> bool:
    """이미 그려 둔 PNG를 그대로 써도 되는지(같은 dpi로 그렸는지) 판단한다."""
    marker = out_dir / _MARKER
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("dpi") == dpi
    except (json.JSONDecodeError, OSError):
        return False
