"""페이지 유형 자동 분류 — 텍스트 레이어 통계만 쓴다.

CER 하나로 뭉뚱그리면 목차 한 장이 평균을 끌고 간다. 유형을 붙여 두면
"본문은 얼마, 표는 얼마"로 갈라 볼 수 있고, 어떤 레이아웃에서 OCR이 무너지는지
모델을 바꿔 가며 같은 기준으로 비교할 수 있다.

분류는 **규칙 4개, 순서대로 처음 걸리는 것**으로 끝낸다. 정교한 레이아웃 분석이
아니라 결과를 묶어 보기 위한 거친 꼬리표다.

    toc        리더(점선) 문자 비율이 높다 → 목차
    table      짧은 줄이 다수 → 표·서식·인사발령 같은 칸 구조
    body_2col  가운데에 글자가 전혀 없는 세로 여백이 있다 → 2단 조판
    body_1col  나머지 (기본값)
"""

from __future__ import annotations

import re
from typing import Any

import pymupdf

#: 리더로 쓰이는 점 문자들. 연속으로 3개 이상 이어질 때만 리더로 센다.
LEADER_CHARS = "·ᆞ•∙⋅・·‧․…‥"

#: 3개 이상 연속된 리더 문자.
LEADER_RUN = re.compile(f"[{LEADER_CHARS}]{{3,}}")

#: 이 비율을 넘으면 목차로 본다.
TOC_LEADER_RATIO = 0.15

#: 이보다 짧은 줄을 "짧은 줄"로 센다(글자 수).
SHORT_LINE_CHARS = 20

#: 짧은 줄이 이 비율을 넘고 줄 수가 충분하면 표로 본다.
TABLE_SHORT_RATIO = 0.60
TABLE_MIN_LINES = 10

#: 단 사이 여백으로 인정할 최소 너비(페이지 폭 대비)와, 여백을 찾을 가운데 구간.
GUTTER_MIN_WIDTH = 0.03
GUTTER_CENTER = (0.30, 0.70)


def leader_ratio(text: str) -> float:
    """전체 글자 중 리더(점 3개 이상 연속)가 차지하는 비율."""
    if not text:
        return 0.0
    return sum(len(m.group()) for m in LEADER_RUN.finditer(text)) / len(text)


def short_line_ratio(text: str) -> tuple[float, int]:
    """짧은 줄의 비율과 전체 줄 수를 돌려준다."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0, 0
    short = sum(1 for ln in lines if len(ln) < SHORT_LINE_CHARS)
    return short / len(lines), len(lines)


def has_gutter(page: pymupdf.Page) -> bool:
    """페이지 가운데에 글자가 전혀 없는 세로 띠가 있으면 2단으로 본다."""
    width = page.rect.width
    if width <= 0:
        return False
    spans = [
        (b[0] / width, b[2] / width)
        for b in page.get_text("blocks")
        if (b[4] or "").strip()
    ]
    if len(spans) < 4:
        return False

    bins = 100
    covered = [False] * bins
    for x0, x1 in spans:
        lo = max(0, int(x0 * bins))
        hi = min(bins - 1, int(x1 * bins))
        for b in range(lo, hi + 1):
            covered[b] = True

    run = 0
    for b in range(bins):
        if covered[b]:
            run = 0
            continue
        run += 1
        start, end = (b - run + 1) / bins, (b + 1) / bins
        center = (start + end) / 2
        if end - start >= GUTTER_MIN_WIDTH and GUTTER_CENTER[0] <= center <= GUTTER_CENTER[1]:
            return True
    return False


def classify_page(page: pymupdf.Page, text: str | None = None) -> str:
    """페이지 하나를 ``toc`` / ``table`` / ``body_2col`` / ``body_1col`` 로 분류한다."""
    body = page.get_text(sort=True) if text is None else text
    if leader_ratio(body) >= TOC_LEADER_RATIO:
        return "toc"
    ratio, n_lines = short_line_ratio(body)
    if n_lines >= TABLE_MIN_LINES and ratio >= TABLE_SHORT_RATIO:
        return "table"
    if has_gutter(page):
        return "body_2col"
    return "body_1col"


def page_stats(page: pymupdf.Page) -> dict[str, Any]:
    """분류 결과와 근거 통계를 함께 돌려준다(결과 JSON 기록용)."""
    body = page.get_text(sort=True)
    ratio, n_lines = short_line_ratio(body)
    return {
        "page_type": classify_page(page, body),
        "leader_ratio": round(leader_ratio(body), 4),
        "short_line_ratio": round(ratio, 4),
        "lines": n_lines,
    }
