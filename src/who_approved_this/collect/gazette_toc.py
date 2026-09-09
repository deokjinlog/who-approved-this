"""관보 목차 파서 — 안건 단위로 자르고 정답 제명을 얻는다.

관보에는 **채점 가능한 정답이 이미 들어 있다.** 1~2페이지 목차가 그 호에 실린
안건을 모두 나열하고, 각 항목에 **공식 제명과 시작 페이지**가 붙어 있다.

    ○ 법률제21884호(인사청문회법 일부개정법률) ·············· 6
    ○ 광주전파관리소고시제2026-11호(무선국(항공국) 호출명칭 변경) ····· 6

시작 페이지가 있으므로 안건별 본문을 잘라낼 수 있고, 그러면
``(본문, 공식 제명)`` 쌍이 자동으로 만들어진다. T5 요약 트랙의 정답이 이것이다.

파싱이 까다로운 지점
    * 발령기관이 종류 앞에 붙는다(``광주전파관리소고시``, ``국토교통부령``).
    * 번호에 하이픈이 들어간다(``제2026-0238호``).
    * 제명 안에 괄호가 중첩된다(``무선국(항공국) 호출명칭 변경``).
    그래서 줄 단위로 자르고 **뒤에서부터** 페이지·괄호를 찾는다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf

#: 목차 항목 줄의 머리표.
BULLET = "○"

#: 목차를 찾을 최대 페이지 수. 관보 목차는 앞쪽 몇 장에 몰려 있다.
TOC_MAX_PAGES = 8

#: ``<발령기관><종류>제<번호>호`` 에서 종류와 번호.
KIND = re.compile(r"(법률|대통령령|총리령|부령|고시|공고|훈령|예규|규칙)\s*제\s*([0-9][0-9\-]*)\s*호")

#: 제명과 페이지 번호 사이를 채우는 점선 리더.
LEADER = re.compile(r"[·․.．]{3,}\s*(\d{1,4})\s*$")


def parse_toc_text(text: str) -> list[dict[str, Any]]:
    """목차 텍스트에서 안건 목록을 뽑는다."""
    items: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(BULLET):
            continue
        line = line.lstrip(BULLET).strip()

        page_match = LEADER.search(line)
        if not page_match:
            continue
        head = line[: page_match.start()].strip()

        kind_match = KIND.search(head)
        if not kind_match:
            continue

        # 제명은 종류·번호 뒤 첫 '(' 부터 줄 끝의 마지막 ')' 까지(괄호 중첩 허용).
        open_at = head.find("(", kind_match.end() - 1)
        close_at = head.rfind(")")
        if open_at < 0 or close_at <= open_at:
            continue

        items.append(
            {
                "kind": kind_match.group(1),
                "no": kind_match.group(2),
                "title": head[open_at + 1 : close_at].strip(),
                "page": int(page_match.group(1)),
                "issuer": head[: kind_match.start(1)].strip(),
            }
        )
    return items


def load_items(pdf_path: Path, max_body_pages: int = 12) -> list[dict[str, Any]]:
    """관보 한 호를 안건 단위로 자른다.

    각 안건의 본문은 **자기 시작 페이지부터 다음 안건 시작 페이지 직전까지**다.
    마지막 안건은 ``max_body_pages`` 로 자른다(뒤에 부록이 붙는 경우가 있다).

    목차의 페이지 번호는 **인쇄 면 번호**이고 PDF 페이지 인덱스와 어긋날 수 있어,
    첫 안건의 인쇄 면 번호와 실제 위치 차이를 재어 전체에 같은 보정을 적용한다.
    """
    with pymupdf.open(pdf_path) as doc:
        toc_text = "".join(
            doc[i].get_text(sort=True) for i in range(min(TOC_MAX_PAGES, doc.page_count))
        )
        items = parse_toc_text(toc_text)
        if not items:
            return []

        offset = _page_offset(doc, items[0])
        out: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            start = item["page"] + offset - 1
            if i + 1 < len(items):
                end = items[i + 1]["page"] + offset - 1
            else:
                end = start + max_body_pages
            start = max(0, min(start, doc.page_count - 1))
            end = max(start + 1, min(end, doc.page_count))
            body = "\n".join(doc[p].get_text(sort=True) for p in range(start, end))
            out.append(
                {
                    **item,
                    "item_id": f"{pdf_path.stem}#{item['kind']}{item['no']}",
                    "source": pdf_path.stem,
                    "pdf_pages": [start + 1, end],
                    "body": body,
                    "body_chars": len(body),
                }
            )
        return out


#: 이만큼 이상 머리표가 있으면 그 페이지는 목차로 본다.
TOC_BULLET_MIN = 3


def _is_toc_page(text: str) -> bool:
    """목차 페이지인지. 항목 머리표가 여러 개 있으면 목차다."""
    return text.count(BULLET) >= TOC_BULLET_MIN


def _page_offset(doc: pymupdf.Document, first: dict[str, Any]) -> int:
    """인쇄 면 번호 → PDF 인덱스 보정값.

    첫 안건의 제명이 실제로 나타나는 페이지를 찾아 목차가 가리킨 번호와의 차이를 쓴다.
    **목차 페이지는 건너뛴다** — 목차에도 같은 제명이 적혀 있어서, 안 걸러내면
    첫 안건이 목차 위치로 잡히고 보정값이 음수가 된다.
    못 찾으면 보정 없음(0)으로 둔다.
    """
    needle = re.sub(r"\s+", "", first["title"])[:12]
    if not needle:
        return 0
    for p in range(min(doc.page_count, first["page"] + 8)):
        text = doc[p].get_text(sort=True)
        if _is_toc_page(text):
            continue
        if needle in re.sub(r"\s+", "", text):
            return (p + 1) - first["page"]
    return 0
