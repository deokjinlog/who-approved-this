"""전결 별표(엑셀) → 구조화된 전결표.

실무에서 위임전결규정 별표는 엑셀 행렬로 다닌다 — 행이 사무, 열이 직급, 칸에 표시.
가평군 사무전결처리규칙 별표1(32시트, 4,164행)이 그 형태다.

    일련번호 | 사 무 명                  | 주무관 | 팀장 | 과장 | 국장 | 부군수 | 군수
    2        | 예산                      |        |      |      |      |        |
             | 가.예산편성자료 및 요구서 제출 | 기안   |      |  ○   |      |        |

    "기안"  → 그 사무를 누가 기안하는지
    "○"    → 누가 전결하는지(최종 결재 직급)

금액 구간은 사무명 안에 글로 들어 있다("5천만원 초과 4억원 이하"). 그런 행은 바로 위의
사무명을 부모로 이어 붙인다.

직급은 기관마다 이름이 달라 **레벨 번호**로 정규화한다::

    0 주무관(담당자)  1 팀장  2 과장·담당관·사업소장  3 국장·직속기관장  4 부군수  5 군수
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import openpyxl

#: 헤더 글자 → 레벨. 긴 것부터 대조한다("부군수"가 "군수"로 잡히지 않게).
LEVEL_KEYS: list[tuple[str, int]] = [
    ("부군수", 4), ("부시장", 4), ("군수", 5), ("시장", 5),
    ("국장", 3), ("직속기관장", 3),
    ("담당관", 2), ("과장", 2), ("사업소장", 2), ("과소장", 2),
    ("팀장", 1), ("주무관", 0), ("담당자", 0), ("담 당 자", 0),
]
LEVEL_NAMES = {0: "주무관", 1: "팀장", 2: "과장", 3: "국장", 4: "부군수", 5: "군수"}

MARKS = {"○", "◯", "O", "ㅇ", "●"}
_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(억|천만|백만|만)\s*원\s*(초과|이상|이하|미만)?")
_UNIT = {"억": 100_000_000, "천만": 10_000_000, "백만": 1_000_000, "만": 10_000}


def parse_amount_range(text: str) -> tuple[int | None, int | None]:
    """"5천만원 초과 4억원 이하" → (50_000_000, 400_000_000). 경계는 초과=배타, 이하=포함."""
    lo = hi = None
    for num, unit, op in _AMOUNT.findall(text):
        value = int(float(num) * _UNIT[unit])
        if op in ("초과", "이상"):
            lo = value
        elif op in ("이하", "미만"):
            hi = value
    return lo, hi


def amount_bands_from_text(item_text: str) -> list[dict[str, Any]]:
    """**자연어 2열형 별표**에서 금액 구간과 전결권자를 뽑는다 (스텁).

    가평군처럼 "사무명 × 직급 ○표" 행렬이면 :func:`parse_delegation_table` 로 충분하다.
    그런데 공기업 규정은 항목 텍스트 안에 금액 기준을 문장으로 넣는 일이 많다
    (조사 §4-2, 한국석유공사)::

        1. 5억원 이상인 경우 사장결재를 득한 후 이사회 의결을 거쳐 시행한다.
        2. 3천만원 이상 5억원 미만인 경우 비축기지건설담당 상임이사의 결재를 득하여 시행한다.

    돌려줄 모양::

        [{"amount_min": 500_000_000, "amount_max": None, "final_title": "사장",
          "extra": "이사회 의결", "text": "1. …"}, …]

    금액 부분은 :func:`parse_amount_range` 를 그대로 쓰면 된다. 남은 일은 번호 조항 분리와
    "○○의 결재를 득하여" 에서 결재 직위를 떼는 것. 실물 별표가 오면 구현한다.
    """
    raise NotImplementedError("자연어 2열형 별표 실물 도착 후 구현 — 금액 부분은 parse_amount_range 사용")


def _header_levels(ws: Any, max_header_rows: int = 8) -> tuple[dict[int, int], int, int]:
    """시트 머리행에서 ``{열번호: 레벨}``, 사무명 열, 데이터 시작 행을 찾는다."""
    col_level: dict[int, int] = {}
    name_col, header_end = 2, 0
    for r in range(1, max_header_rows + 1):
        for c in range(1, min(ws.max_column, 12) + 1):
            v = ws.cell(r, c).value
            if not isinstance(v, str):
                continue
            flat = re.sub(r"\s+", "", v)
            if "사무명" in flat:
                name_col, header_end = c, max(header_end, r)
            for key, level in LEVEL_KEYS:
                if re.sub(r"\s+", "", key) in flat:
                    col_level.setdefault(c, level)
                    header_end = max(header_end, r)
                    break
    return col_level, name_col, header_end + 1


def parse_delegation_table(path: Path) -> list[dict[str, Any]]:
    """별표 엑셀 전체를 ``[{sheet, group, item, text, amount_min, amount_max,
    drafter_level, final_level, final_title}]`` 로 편다."""
    wb = openpyxl.load_workbook(path, data_only=True)
    out: list[dict[str, Any]] = []
    for ws in wb.worksheets:
        col_level, name_col, start = _header_levels(ws)
        if not col_level:
            continue
        group = item = ""
        for r in range(start, ws.max_row + 1):
            text = ws.cell(r, name_col).value
            text = re.sub(r"\s+", " ", str(text)).strip() if text else ""
            serial = ws.cell(r, name_col - 1).value if name_col > 1 else None
            marks = {c: str(ws.cell(r, c).value or "").strip() for c in col_level}
            final = [col_level[c] for c, v in marks.items() if v in MARKS]
            drafter = [col_level[c] for c, v in marks.items() if "기안" in v]
            if not text:
                continue
            if serial not in (None, "") and not final:
                group, item = text, ""          # "1 | 의회관계" 같은 묶음 머리
                continue
            lo, hi = parse_amount_range(text)
            is_amount_row = (lo is not None or hi is not None) and len(text) < 40
            if not final:
                item = text                     # 금액 행의 부모가 될 사무명
                continue
            full = f"{item} {text}".strip() if is_amount_row and item else text
            if not is_amount_row:
                item = text
            level = max(final)                  # ○ 가 여럿이면 가장 높은 직급
            out.append({
                "sheet": ws.title.strip(),
                "group": group,
                "item": full,
                "amount_min": lo,
                "amount_max": hi,
                "drafter_level": min(drafter) if drafter else None,
                "final_level": level,
                "final_title": LEVEL_NAMES[level],
            })
    return out
