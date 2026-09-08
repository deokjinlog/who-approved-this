"""T1 규칙 기반 필드 추출 — pymupdf 텍스트 레이어 + 블록 좌표.

이 파서는 **상한선을 재는 자리**다. 텍스트 레이어가 살아 있는 PDF에서
규칙만으로 어디까지 뽑히는지 먼저 보고, 그 위에 OCR·VLM을 얹어 비교한다.

양식 의존성
-----------
아래 규칙은 대한민국 행정기관 공문서 양식에 기댄다. 다른 양식(사내 품의서 등)에는
그대로 쓰이지 않는다. 양식에 기댄 부분은 각 함수에 ``양식 의존:`` 으로 표시했다.

결재란(T1b)의 현실
-------------------
10건 진단 결과 결재란이 텍스트로 잡히는 건 **2건뿐**이었다. 나머지는 상단 결재란
영역에 직위 문자열이 없다(이미지이거나 공개본에서 빠진 것으로 보인다).
그래서 :meth:`FieldTextLayerParser.parse` 는 결재선을 못 찾아도 예외를 내지 않고
빈 목록을 돌려주며, 그 사실 자체를 결과에 남긴다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf

from who_approved_this.parse.base import Parser

#: 결재란에 나타나는 직위. 긴 것부터 찾아야 "주사보"가 "주사"로 잘리지 않는다.
TITLES = (
    "기록연구사", "본부장", "센터장", "담당관", "주사보", "연구관", "연구사",
    "서기관", "사무관", "주무관", "지사장", "장관", "차관", "실장", "국장",
    "과장", "팀장", "관장", "원장", "소장", "부장", "계장", "차장", "주사", "서기",
)

#: 결재란 위쪽에 붙는 역할 표시.
ROLE_MARKS = ("전결", "대결", "협조")

#: "총무과-7941" 같은 문서번호.
#: 하이픈 **바로 앞의 한글 덩어리**만 부서명으로 잡는다. 공백을 허용하면
#: 앞 문장까지 삼켜 "정답보다 김" 불일치가 난다.
DOC_NO = re.compile(r"([가-힣]{2,12})\s*-\s*(\d{1,7})")

#: "2026. 9. 7." / "2026-09-07" 형태의 날짜.
DATE = re.compile(r"(\d{4})\s*[.\-]\s*(\d{1,2})\s*[.\-]\s*(\d{1,2})")

#: 제목 줄. 양식 의존: 공문서는 "제목" 라벨을 쓴다.
SUBJECT = re.compile(r"제\s*목\s*[:：]?\s*(.+)")

#: 결재란으로 볼 페이지 상단 영역(포인트). 양식 의존: A4 공문서 기준.
APPROVAL_TOP = 200.0


def _first(pattern: re.Pattern[str], text: str, group: int = 1) -> str:
    m = pattern.search(text)
    return m.group(group).strip() if m else ""


def extract_doc_no(text: str) -> str:
    """문서번호를 뽑는다. 양식 의존: "<부서명>-<일련번호>" 형태."""
    m = DOC_NO.search(text)
    return f"{m.group(1).strip()}-{int(m.group(2))}" if m else ""


def extract_dept(text: str, doc_no: str) -> str:
    """담당부서명. 양식 의존: 문서번호 앞부분이 곧 부서명이다."""
    return doc_no.split("-")[0].strip() if "-" in doc_no else ""


def extract_date(text: str) -> str:
    """생산일자를 ``YYYY-MM-DD`` 로 맞춘다.

    양식 의존: 공문서에는 결재일·시행일·접수일이 함께 찍힌다. 포털의 ``생산일자``는
    그중 **가장 이른 날짜**(기안·결재일)와 맞는 경우가 많아 최솟값을 쓴다.
    """
    found = [
        f"{y}-{int(mo):02d}-{int(d):02d}" for y, mo, d in DATE.findall(text)
    ]
    return min(found) if found else ""


def extract_subject(text: str) -> str:
    """제목. 양식 의존: "제목" 라벨 뒤 한 줄."""
    return _first(SUBJECT, text).splitlines()[0].strip() if SUBJECT.search(text) else ""


def extract_org(text: str) -> str:
    """생산기관명.

    양식 의존: 발신명의가 페이지 하단에 "○○부장관" / "○○도지사" 처럼 찍힌다.
    공백을 넣어 자간을 벌리는 경우가 많아(``법 무 부 장 관``) 공백을 걷고 찾는다.
    """
    flat = re.sub(r"\s+", "", text)
    m = re.search(r"([가-힣]{2,12}?(?:부|청|처|위원회|도|시|군|구|원|공사|공단))(?:장관|장|지사|시장|군수|구청장|위원장)", flat)
    return m.group(1) if m else ""


#: 직위 칸 아래에서 이름을 찾을 때 허용하는 가로 어긋남(포인트).
CELL_X_TOLERANCE = 30.0

#: 직위 칸 아래 이 거리 안의 글자만 같은 칸으로 본다(포인트).
CELL_Y_REACH = 40.0


def _approval_words(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    """결재란 영역의 단어를 ``(중심x, y, 단어)`` 로 돌려준다."""
    return [
        ((w[0] + w[2]) / 2, w[1], w[4].strip())
        for w in page.get_text("words")
        if w[1] < APPROVAL_TOP and w[4].strip()
    ]


def _name_below(words: list[tuple[float, float, str]], x: float, y: float) -> str:
    """``(x, y)`` 직위 칸 **바로 아래**의 이름을 찾는다.

    양식 의존: 결재란은 표라서 직위와 이름이 다른 줄에 있고 같은 열에 세로로 쌓인다.
    글자 순서로 이어 읽으면 옆 칸 직위가 먼저 걸리므로 좌표로 짝지어야 한다.
    """
    below = [
        (wy, word)
        for wx, wy, word in words
        if y < wy <= y + CELL_Y_REACH and abs(wx - x) <= CELL_X_TOLERANCE
    ]
    for _wy, word in sorted(below):
        name = re.sub(r"[^가-힣]", "", word)
        if 2 <= len(name) <= 4 and name not in TITLES:
            return name
    return ""


def extract_approval_line(page: pymupdf.Page) -> list[dict[str, str]]:
    """결재란에서 ``[{"role", "title", "name"}]`` 를 뽑는다.

    양식 의존이 가장 큰 부분이다. 공문서 결재란은 페이지 **상단의 표**이고,
    왼쪽부터 기안 → 검토 → 결재 순으로 칸이 놓인다. 그래서

    1. 상단 영역(:data:`APPROVAL_TOP` 위쪽) 단어만 본다.
    2. 직위 단어를 x 좌표 순으로 세운다.
    3. 각 직위 **바로 아래 칸**에서 이름을 찾는다(:func:`_name_below`).
    4. 왼쪽 끝을 ``기안``, 오른쪽 끝을 ``결재``, 가운데를 ``검토`` 로 본다.
       칸이 하나뿐이면 ``결재`` 로 본다.
    5. 같은 영역에 ``전결``/``대결``/``협조`` 표시가 있으면 그 역할을 우선한다.

    결재란이 이미지인 문서에서는 빈 목록이 나오며, 그건 실패가 아니라 **측정 대상**이다.
    """
    words = _approval_words(page)
    marks = {m for _x, _y, w in words for m in ROLE_MARKS if m in w}

    cells: list[tuple[float, str, str]] = []
    for x, y, word in words:
        title = next((t for t in TITLES if t in word), "")
        if title:
            cells.append((x, title, _name_below(words, x, y)))

    cells.sort(key=lambda c: c[0])
    out: list[dict[str, str]] = []
    for i, (_x, title, name) in enumerate(cells):
        if "협조" in marks and len(cells) == 1:
            role = "협조"
        elif len(cells) == 1 or i == len(cells) - 1:
            role = "결재"
        elif i == 0:
            role = "기안"
        else:
            role = "검토"
        out.append({"role": role, "title": title, "name": name})
    return out


def extract_person(page: pymupdf.Page) -> str:
    """담당자(기안자) 이름 — 결재란 **가장 왼쪽 칸**의 이름.

    양식 의존: 결재란은 왼쪽이 기안자다. 포털 메타데이터의 ``담당자`` 도 기안자다.
    """
    line = extract_approval_line(page)
    return next((c["name"] for c in line if c["role"] == "기안"), "") or next(
        (c["name"] for c in line if c["name"]), ""
    )


class FieldTextLayerParser(Parser):
    """PDF 한 건에서 T1a 6필드와 T1b 결재선을 규칙으로 뽑는다.

    입력 ``pdf_path: Path`` → 출력 ``dict``::

        {"제목", "생산기관", "담당부서", "담당자", "생산일자", "문서번호",
         "approval_line": [{"role", "title", "name"}, ...]}

    ``담당자``(기안자 실명)는 결재란에서 기안 역할의 이름을 가져온다.
    값은 평가 단계까지만 흐르고 결과 JSON에는 남지 않는다.
    """

    name = "textlayer-rules"

    def parse(self, pdf_path: Path) -> dict[str, Any]:
        """``pdf_path`` 첫 페이지에서 필드와 결재선을 뽑는다."""
        with pymupdf.open(pdf_path) as doc:
            page = doc[0]
            text = page.get_text(sort=True)
            line = extract_approval_line(page)
            drafter = extract_person(page)

        doc_no = extract_doc_no(text)
        return {
            "제목": extract_subject(text),
            "생산기관": extract_org(text),
            "담당부서": extract_dept(text, doc_no),
            "담당자": drafter,
            "생산일자": extract_date(text),
            "문서번호": doc_no,
            "approval_line": line,
        }
