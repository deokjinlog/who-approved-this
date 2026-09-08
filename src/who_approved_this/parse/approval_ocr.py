"""결재선 추출 — 결재란 영역에서 ``직위·이름`` 칸을 뽑는다.

같은 함수가 두 입력을 다룬다.

* ``source="ocr"`` — 결재란을 잘라 200dpi로 렌더링하고 Apple Vision(ocrmac)으로 읽는다.
* ``source="textlayer"`` — 같은 영역의 pymupdf 텍스트 레이어 단어를 그대로 쓴다.

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

from who_approved_this.parse.base import Parser
from who_approved_this.parse.region_crop import TITLE_HINTS, crop_approval

#: 이름을 찾을 세로 거리(원본 페이지 포인트). 시행문형은 직위 바로 아래 줄이지만,
#: 격자형은 표 머리행과 이름행 사이가 훨씬 넓어 영역 종류별로 다르게 잡는다.
NAME_Y_REACH = {"bottom": 26.0, "top": 90.0, "override": 90.0}

#: 이름을 찾을 가로 어긋남(포인트).
NAME_X_TOLERANCE = 70.0

#: 두 줄로 접힌 직위를 잇기 위한 세로 거리.
TITLE_WRAP_Y = 16.0

ROLE_COOP = "협조자"


def _words_textlayer(page: pymupdf.Page, rect: pymupdf.Rect) -> list[tuple[float, float, str]]:
    """텍스트 레이어 단어를 ``(x중심, y상단, 글자)`` 로."""
    return [
        ((w[0] + w[2]) / 2, w[1], w[4].strip())
        for w in page.get_text("words")
        if pymupdf.Rect(w[:4]).intersects(rect) and w[4].strip()
    ]


def _words_ocr(png: Path, rect: pymupdf.Rect) -> list[tuple[float, float, str]]:
    """OCR 결과를 원본 페이지 좌표로 되돌려 ``(x중심, y상단, 글자)`` 로.

    ocrmac 좌표는 잘라낸 이미지 기준의 **좌하단 원점 정규화 좌표**다.
    """
    out: list[tuple[float, float, str]] = []
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
            frac = (offset + len(token) / 2) / n_chars
            out.append((rect.x0 + (x + w * frac) * rect.width, top, token))
            offset += len(token)
    return out


def _merge_wrapped_titles(words: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """두 줄로 접힌 직위를 잇는다("출입국관리서" + "기" → "출입국관리서기")."""
    merged: list[tuple[float, float, str]] = []
    used: set[int] = set()
    for i, (x, y, t) in enumerate(sorted(words, key=lambda w: (w[1], w[0]))):
        if i in used:
            continue
        for j, (x2, y2, t2) in enumerate(sorted(words, key=lambda w: (w[1], w[0]))):
            if j <= i or j in used:
                continue
            if not (0 < y2 - y <= TITLE_WRAP_Y and abs(x2 - x) <= NAME_X_TOLERANCE):
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
                merged.append((x, y2, joined))
                used.add(j)
                break
        else:
            merged.append((x, y, t))
    return merged


def extract_cells(
    words: list[tuple[float, float, str]], region: str = "bottom"
) -> list[dict[str, str]]:
    """단어 목록에서 ``[{"role", "title", "name"}]`` 를 뽑는다."""
    reach = NAME_Y_REACH.get(region, 26.0)
    words = _merge_wrapped_titles(words)
    coop_y = next((y for _x, y, t in words if ROLE_COOP in t), None)

    cells: list[tuple[float, float, str, str]] = []
    for x, y, token in words:
        if not any(h in token for h in TITLE_HINTS):
            continue
        name = ""
        best = reach + 1
        for x2, y2, t2 in words:
            gap = y2 - y
            if 0 < gap <= reach and abs(x2 - x) <= NAME_X_TOLERANCE and gap < best:
                cand = "".join(c for c in t2 if "가" <= c <= "힣")
                if 2 <= len(cand) <= 4 and not any(h in cand for h in TITLE_HINTS):
                    name, best = cand, gap
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
        else:
            with pymupdf.open(pdf_path) as doc:
                words = _words_textlayer(doc[0], rect)
        return {
            "source": self.source,
            "region": kind,
            "cells": extract_cells(words, kind),
        }
