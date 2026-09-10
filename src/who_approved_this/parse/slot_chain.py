"""슬롯 체인 결재선 — 조직도(부서→국) + 직무대리 표.

가평군 gold 30건은 전부 같은 뼈대였다::

    기안 → 팀장 → 과장(담당관·소장) → 국장(국 소속일 때) → 부군수 → 군수

문서마다 달라 보인 건 **슬롯을 채운 사람**이다. 7월부터는 부군수 칸에 행정안전국장이,
7월까지는 건설도시국장 칸에 건설과장이 찍혔다. 이건 전결 규정이 아니라 인사 데이터다.
그래서 두 층으로 나눈다.

    ① 슬롯 체인   조직도만으로 만든다 (결정론)
    ② 직무대리    날짜별 대리 표로 칸 직위를 바꾼다 (인사 데이터)

같은 직위가 연달아 나오면 한 칸으로 합친다. 행정안전국 소속 과의 문서는 국장 슬롯과
부군수 슬롯을 둘 다 행정안전국장이 채우니 한 칸이 된다(gold #12 #16 #19 등).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LEVELS = {"주무관": 0, "팀장": 1, "과장": 2, "국장": 3, "부군수": 4, "군수": 5}


def title_level(title: str) -> int:
    """직위 글자로 레벨을 짐작한다. 조직도에 없는 직위(TF팀장 등)도 받는다."""
    t = re.sub(r"\s+", "", title)
    if t.endswith("부군수"):
        return 4
    if t.endswith("군수"):
        return 5
    if t.endswith("국장"):
        return 3
    if t.endswith(("과장", "담당관", "사업소장")):
        return 2
    if t.endswith(("팀장", "소장")):
        return 1
    return 0


@dataclass
class OrgChart:
    """``org_chart.tsv`` (기관·부서·팀·직위·담당업무·전화번호). 이름 열은 없다."""

    bureau: dict[str, str | None] = field(default_factory=dict)   # 부서 → 국 (직속이면 None)
    head: dict[str, str] = field(default_factory=dict)            # 부서 → 부서장 직위
    teams: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    # teams[부서][팀] = {"lead": 팀장 직위, "duties": 담당업무 이어붙인 글}

    @classmethod
    def load(cls, path: Path) -> "OrgChart":
        org = cls()
        with path.expanduser().open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        for r in rows:
            parent, unit, title = r["부서"].strip(), r["팀"].strip(), r["직위"].strip()
            # 부서 행: 상위가 "OO국" 이거나 기관 자신(직속)
            if parent.endswith("국") or parent == r["기관"].strip():
                org.bureau.setdefault(unit, parent if parent.endswith("국") else None)
                if title and title_level(title) == 2:
                    org.head.setdefault(unit, title)
                continue
            team = org.teams.setdefault(parent, {}).setdefault(
                unit, {"lead": "", "duties": ""}
            )
            if title.endswith("팀장") or (title.endswith("소장") and not team["lead"]):
                team["lead"] = title
            team["duties"] += " " + (r.get("담당업무") or "")
        return org

    def team_of_lead(self, dept: str, lead: str) -> str | None:
        norm = re.sub(r"\s+", "", lead)
        for name, t in self.teams.get(dept, {}).items():
            if re.sub(r"\s+", "", t["lead"]) == norm:
                return name
        return None


def _in_range(date: str | None, start: str | None, end: str | None) -> bool:
    if date is None:
        return False
    return (start is None or date >= start) and (end is None or date <= end)


def build_slot_chain(
    dept: str,
    drafter_title: str,
    team_lead: str | None,
    org: OrgChart,
    cfg: dict[str, Any],
    *,
    date: str | None = None,
    final_level: int = 5,
    use_acting: bool = True,
) -> list[dict[str, str]]:
    """결재선 칸 목록을 만든다. 칸마다 ``slot`` 과 ``evidence`` 를 단다.

    ``drafter_title`` 과 ``team_lead`` 는 **입력**이다(로그인 사용자와 그 소속 팀).
    ``final_level`` 은 전결 조회 결과 — 그 레벨까지만 올라간다.
    """
    head = org.head.get(dept, f"{dept}장")
    bureau = org.bureau.get(dept)
    skip_deputy = {s["dept"] for s in cfg.get("chain", {}).get("skip_deputy", [])}

    slots: list[dict[str, Any]] = [
        {"slot": "기안", "title": drafter_title, "level": title_level(drafter_title),
         "evidence": "input"}
    ]
    if team_lead and team_lead != drafter_title:
        slots.append({"slot": "팀장", "title": team_lead, "level": 1, "evidence": "input:team"})
    if head != drafter_title:
        slots.append({"slot": "과장", "title": head, "level": 2, "evidence": "org_chart"})
    if bureau:
        slots.append({"slot": f"국장:{bureau}", "title": f"{bureau}장", "level": 3,
                      "evidence": "org_chart"})
    if dept not in skip_deputy:
        slots.append({"slot": "부군수", "title": "부군수", "level": 4, "evidence": "org_chart"})
    slots.append({"slot": "군수", "title": "군수", "level": 5, "evidence": "org_chart"})

    # 전결 — 최종 결재 레벨 위는 자른다. 기안 칸은 항상 남긴다.
    slots = [slots[0]] + [s for s in slots[1:] if s["level"] <= final_level]

    if use_acting:
        for a in cfg.get("acting", []):
            if not _in_range(date, a.get("from"), a.get("to")):
                continue
            for s in slots:
                if s["slot"] != a["slot"]:
                    continue
                if a.get("mode") == "annotate":
                    # 권한대행: 칸 직위는 권한 주체 그대로, 서명자만 대행.
                    # 대행자 본인 칸은 따로 두지 않는다(gold #11 #22 에 부군수 칸 없음).
                    s["acting_by"] = a["by"]
                    s["evidence"] += f"+acting:{a['by']}"
                    slots = [x for x in slots if x is s or x["title"] != a["by"]]
                else:
                    s["title"] = a["by"]
                    s["evidence"] = f"acting:{a['slot']}"

    # 같은 직위가 연달아 나오면 한 칸 (국장 슬롯과 부군수 슬롯을 한 사람이 채운 경우)
    merged: list[dict[str, Any]] = []
    for s in slots:
        if merged and re.sub(r"\s+", "", merged[-1]["title"]) == re.sub(r"\s+", "", s["title"]):
            merged[-1]["slot"] += "+" + s["slot"]
            continue
        merged.append(dict(s))

    for i, s in enumerate(merged):
        s["role"] = "기안" if i == 0 else ("결재" if i == len(merged) - 1 else "검토")
    return merged
