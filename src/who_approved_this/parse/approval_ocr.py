"""결재선 추출 — 결재란 영역에서 ``직위·이름`` 칸을 뽑는다.

같은 함수가 두 입력을 다룬다.

* ``source="textlayer"`` — 같은 영역의 pymupdf 텍스트 레이어 단어를 그대로 쓴다.
* ``source="ocr"`` — 결재란을 잘라 200dpi로 렌더링하고 Apple Vision(ocrmac)으로 읽는다.
* ``source="paddle"`` — 같은 이미지를 PaddleOCR(PP-OCRv5, korean)로 읽는다.

  Vision 은 표의 여러 칸을 한 관측으로 묶어 주는 일이 잦아 칸별 x 좌표가 뭉개진다.
  PaddleOCR 은 줄(칸) 단위로 따로 검출해 칸 구조가 그대로 남는다. 대신 모델을
  내려받아야 하고(최초 1회, ``~/.paddlex``) 느리다. 원문은 로컬을 벗어나지 않는다.

칸 짜맞추기 규칙 (양식 의존)
----------------------------
10건을 눈으로 확인한 결과 두 양식 모두 **직위가 위, 이름이 바로 아래**에 온다.
시행문형은 ``주무관 / (아래) 허○○`` 이 가로로 반복되고, 격자형은 표 머리행이 직위,
그 아래 행이 이름이다. 그래서

1. 영역 안 단어를 ``(x, y, 글자)`` 로 모은다.
2. 직위처럼 보이는 단어를 찾는다(:data:`TITLE_HINTS` 부분일치).
   두 줄로 접힌 직위("출입국관리서" + "기")는 같은 x 열의 윗줄과 이어 붙인다.
3. 각 직위 **아래쪽 가까운 칸**에서 한글 2~4자를 이름으로 집는다.
4. ``협조자`` 표시 오른쪽/아래의 칸은 역할을 ``협조`` 로 둔다.
5. 나머지는 왼쪽부터 ``기안`` → ``검토``… → 마지막 ``결재``.

역할 이름은 문서에 **적혀 있지 않다**(직위와 이름만 인쇄된다). 위치로 추정하는
것이므로 gold 도 같은 규칙으로 만들었고, 이 추정이 틀리는 것 자체가 측정 대상이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
from ocrmac import ocrmac
from PIL import Image

from who_approved_this.parse.base import Parser
from who_approved_this.parse.region_crop import TITLE_HINTS, crop_approval

#: 직위 대비 이름이 놓이는 자리. 양식마다 다르므로 영역 종류별로 창을 따로 준다.
#: ``(dx_min, dx_max, dy_min, dy_max)`` — 원본 페이지 포인트, y는 아래가 +.
#:
#: * ``bottom`` 시행문형 — 이름이 직위 **오른쪽 같은 줄**에 온다("주무관  허○○").
#:   OCR은 두 글자의 위쪽 모서리를 거의 같게 잡으므로 dy 아래끝을 음수까지 연다.
#: * ``top`` 격자형 — 이름이 직위 **바로 아래 칸**, 같은 열에 온다.
PAIR_WINDOW = {
    "bottom": (0.0, 95.0, -8.0, 30.0),
    "top": (-45.0, 45.0, 8.0, 95.0),
    "override": (-45.0, 95.0, -8.0, 95.0),
}

#: 두 줄로 접힌 직위를 이을 때의 가로 어긋남 상한.
#: 양식마다 정렬이 다르다 — 시행문형은 **왼쪽 정렬**이라 왼쪽 모서리가 맞고,
#: 격자형 표는 칸 안에서 **가운데 정렬**이라 중심이 맞는다. 둘 중 하나만 보면
#: 다른 양식이 깨지므로 두 기준 중 하나라도 맞으면 같은 열로 본다.
#: 값은 격자형 열 간격(약 50pt)보다 작아야 옆 칸을 끌어오지 않는다.
TITLE_WRAP_LEFT_X = 14.0
TITLE_WRAP_CENTER_X = 22.0

#: 두 줄로 접힌 직위를 잇기 위한 세로 거리.
#: 격자형 표의 머리행 두 줄 간격이 18pt 라 그보다 넉넉히 잡는다.
TITLE_WRAP_Y = 26.0

ROLE_COOP = "협조자"

#: 단어 하나: ``(x중심, y상단, 글자, x왼쪽)``.
Word = tuple[float, float, str, float]


def _words_textlayer(page: pymupdf.Page, rect: pymupdf.Rect) -> list[Word]:
    """텍스트 레이어 단어를 ``(x중심, y상단, 글자, x왼쪽)`` 로."""
    return [
        ((w[0] + w[2]) / 2, w[1], w[4].strip(), w[0])
        for w in page.get_text("words")
        if pymupdf.Rect(w[:4]).intersects(rect) and w[4].strip()
    ]


def _words_ocr(png: Path, rect: pymupdf.Rect) -> list[Word]:
    """OCR 결과를 원본 페이지 좌표로 되돌려 ``(x중심, y상단, 글자)`` 로.

    ocrmac 좌표는 잘라낸 이미지 기준의 **좌하단 원점 정규화 좌표**다.
    """
    out: list[Word] = []
    for text, _conf, (x, y, w, h) in ocrmac.OCR(
        str(png), recognition_level="accurate", language_preference=["ko-KR", "en-US"]
    ).recognize():
        top = rect.y0 + (1.0 - (y + h)) * rect.height
        # Vision 은 표의 여러 칸을 한 관측으로 묶어 주기도 한다
        # (예: "환경연구 환경연구 생물소재 생물자원" = 칸 4개).
        # 관측 하나에 x 를 하나만 주면 칸이 겹쳐 짝짓기가 깨지므로,
        # 글자 오프셋 비율로 토큰마다 x 를 나눠 준다.
        n_chars = max(len(text), 1)
        offset = 0
        for token in text.split():
            offset = text.find(token, offset)
            mid = (offset + len(token) / 2) / n_chars
            left = offset / n_chars
            out.append(
                (
                    rect.x0 + (x + w * mid) * rect.width,
                    top,
                    token,
                    rect.x0 + (x + w * left) * rect.width,
                )
            )
            offset += len(token)
    return out


def _merge_wrapped_titles(words: list[Word]) -> list[Word]:
    """두 줄로 접힌 직위를 잇는다("출입국관리서" + "기" → "출입국관리서기")."""
    merged: list[Word] = []
    used: set[int] = set()
    ordered = sorted(words, key=lambda w: (w[1], w[0]))
    for i, (x, y, t, x0) in enumerate(ordered):
        if i in used:
            continue
        for j, (x2, y2, t2, x0b) in enumerate(ordered):
            if j <= i or j in used:
                continue
            same_column = (
                abs(x0b - x0) <= TITLE_WRAP_LEFT_X  # 왼쪽 정렬(시행문형)
                or abs(x2 - x) <= TITLE_WRAP_CENTER_X  # 가운데 정렬(격자형)
            )
            if not (0 < y2 - y <= TITLE_WRAP_Y and same_column):
                continue
            # 윗줄이 이미 완성된 직위면(예: "총무사무관") 아랫줄은 이름이므로 잇지 않는다.
            # 윗줄이 2~3자면 사람 이름일 가능성이 커서(예: "정유미" + "주무관") 잇지 않는다.
            # 한글로만 이뤄진 조각끼리만 잇는다(날짜 "2026." 등이 붙는 걸 막는다).
            hangul = lambda w: bool(w) and all("가" <= c <= "힣" for c in w)
            if any(h in t for h in TITLE_HINTS) or 2 <= len(t) <= 3:
                continue
            if not (hangul(t) and hangul(t2)):
                continue
            joined = t + t2
            if any(h in joined for h in TITLE_HINTS) and len(t2) <= 5:
                # 접힌 직위의 위치는 **아랫줄** 기준으로 잡아야 다른 칸과 줄이 맞는다.
                merged.append((x, y2, joined, x0))
                used.add(j)
                break
        else:
            merged.append((x, y, t, x0))
    return merged


def extract_cells(words: list[Word], region: str = "bottom") -> list[dict[str, str]]:
    """단어 목록에서 ``[{"role", "title", "name"}]`` 를 뽑는다."""
    words = _merge_wrapped_titles(words)
    coop_y = next((y for _x, y, t, _x0 in words if ROLE_COOP in t), None)

    dx_min, dx_max, dy_min, dy_max = PAIR_WINDOW.get(region, PAIR_WINDOW["bottom"])

    titles = [(x, y, t) for x, y, t, _x0 in words if any(h in t for h in TITLE_HINTS)]
    used_names: set[int] = set()
    cells: list[tuple[float, float, str, str]] = []
    for x, y, token in titles:
        name, best, pick = "", None, None
        for k, (x2, y2, t2, _x0b) in enumerate(words):
            if k in used_names:
                continue
            dx, dy = x2 - x, y2 - y
            if not (dx_min <= dx <= dx_max and dy_min <= dy <= dy_max):
                continue
            cand = "".join(c for c in t2 if "가" <= c <= "힣")
            if not (2 <= len(cand) <= 4) or any(h in cand for h in TITLE_HINTS):
                continue
            dist = abs(dx) + abs(dy) * 2  # 같은 열/줄을 우선한다
            if best is None or dist < best:
                name, best, pick = cand, dist, k
        if pick is not None:
            used_names.add(pick)  # 한 이름이 두 직위에 붙지 않게 한다
        cells.append((y, x, token, name))

    cells.sort(key=lambda c: (round(c[0] / 12), c[1]))
    approve = [c for c in cells if coop_y is None or c[0] < coop_y]
    coop = [c for c in cells if coop_y is not None and c[0] >= coop_y]

    out: list[dict[str, str]] = []
    for i, (_y, _x, title, name) in enumerate(approve):
        role = "기안" if i == 0 else ("결재" if i == len(approve) - 1 else "검토")
        out.append({"role": role, "title": title, "name": name})
    out += [{"role": "협조", "title": t, "name": n} for _y, _x, t, n in coop]
    return out


#: PaddleOCR 인스턴스는 만들 때 모델을 올리므로 한 번만 만들어 재사용한다.
_PADDLE: Any = None


def _paddle_engine() -> Any:
    """PaddleOCR(PP-OCRv5, korean) 엔진을 지연 생성한다."""
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR

        _PADDLE = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _PADDLE


def _words_paddle(png: Path, rect: pymupdf.Rect) -> list[Word]:
    """PaddleOCR 결과를 원본 페이지 좌표로 되돌린다.

    좌표는 **잘라낸 이미지의 픽셀** 이므로 이미지 폭·높이로 나눠 비율로 바꾼 뒤
    영역 좌표에 얹는다. 크기는 반드시 PIL 로 잰다 — pymupdf 로 PNG를 열면
    페이지 크기를 72dpi 포인트로 돌려주므로(200dpi 이미지면 2.78배 어긋난다)
    좌표가 통째로 틀어져 이름 짝짓기가 전멸한다.
    """
    result = _paddle_engine().predict(str(png))
    if not result:
        return []
    res = result[0]
    with Image.open(png) as img:
        iw, ih = img.size

    out: list[Word] = []
    for text, poly in zip(res["rec_texts"], res["rec_polys"]):
        token = text.strip()
        if not token:
            continue
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        left = rect.x0 + (min(xs) / iw) * rect.width
        right = rect.x0 + (max(xs) / iw) * rect.width
        top = rect.y0 + (min(ys) / ih) * rect.height
        out.append(((left + right) / 2, top, token, left))
    return out


class ApprovalLineParser(Parser):
    """결재란에서 결재선을 뽑는다. OCR / 텍스트 레이어 중 하나를 고른다.

    입력 ``pdf_path: Path`` → 출력 ``dict``::

        {"source": "ocr"|"textlayer", "region": "top"|"bottom",
         "cells": [{"role", "title", "name"}, ...]}
    """

    def __init__(
        self,
        crop_dir: Path,
        source: str = "ocr",
        dpi: int = 200,
        overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.crop_dir = crop_dir
        self.source = source
        self.dpi = dpi
        self.overrides = overrides or {}
        self.name = f"approval-{source}@{dpi}dpi"

    def parse(self, pdf_path: Path) -> dict[str, Any]:
        """``pdf_path`` 1페이지 결재란에서 결재선을 뽑는다."""
        png, rect, kind = crop_approval(pdf_path, self.crop_dir, self.dpi, self.overrides)
        if self.source == "ocr":
            words = _words_ocr(png, rect)
        elif self.source == "paddle":
            words = _words_paddle(png, rect)
        else:
            with pymupdf.open(pdf_path) as doc:
                words = _words_textlayer(doc[0], rect)
        return {
            "source": self.source,
            "region": kind,
            "cells": extract_cells(words, kind),
        }
