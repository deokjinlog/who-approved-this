"""공식 전결 규정 결재선 — 규정·조직도·직무대리로 결재선을 만들고, 모르는 건 모른다고 한다.

``rules/gapyeong.yaml`` 한 기관용. T6 평가(``wat run t6``)와 API·에이전트가 같은 코드를 쓴다.

    입력   부서, 기안자 직위, 소속 팀(로그인 사용자), 제목, 본문, 날짜
    ①     종점 — 집행 품의면 금액 구간표(결정론), 아니면 별표 행 검색 + LLM 선택
    ②     경로 — 조직도 슬롯 체인 + 직무대리 (결정론)
    출력   칸마다 evidence, 그리고 **확인 필요** 목록

T6 실측으로 정한 기본값: BM25 후보 5개 + LLM(생각 끔). 후보를 늘리면 8B 모델 선택이
무너졌고(9/19), 생각을 켜면 회복하지만 문서당 24초였다.
"""

from __future__ import annotations

import re
from datetime import date as _date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from who_approved_this.parse.delegation_lookup import DelegationLookup, load_table
from who_approved_this.parse.slot_chain import OrgChart, build_slot_chain, title_level
from who_approved_this.tracks.t1c_predictors import _THINK_BLOCK, ollama_chat

RULES = Path(__file__).resolve().parents[3] / "rules" / "gapyeong.yaml"
LEVEL_NAMES = {0: "주무관", 1: "팀장", 2: "과장", 3: "국장", 4: "부군수", 5: "군수"}

PICK_SYSTEM = (
    "너는 지방자치단체 사무전결 규정 담당자다. 문서가 전결대상업무표의 어느 사무에 "
    "해당하는지 후보 기호(A, B, C …) 하나만 답한다. 설명하지 않는다."
)
LETTERS = "ABCDEFGHIJKLMNOP"
#: 별표 사무명 앞의 자체 번호("16. ", "가.", "○"). 후보 번호와 섞이면 모델이
#: 사무명 속 번호를 답한다(하수도 "1. 하수도 특별회계…" → "1").
_ITEM_ENUM = re.compile(r"^\s*(?:\d+\.|[가-하]\.|○|◯|-)\s*")


def pick_row(
    model: str, title: str, body: str, cands: list[dict[str, Any]], think: bool = False
) -> tuple[int, str]:
    """후보 중 하나를 고르게 한다. (고른 순번, 모델 원답) — 원답은 진단용으로 남긴다."""
    listing = "\n".join(
        f"{LETTERS[i]}) [{_ITEM_ENUM.sub('', c['group'])}] {_ITEM_ENUM.sub('', c['item'])}"
        for i, c in enumerate(cands)
    )
    user = f"제목: {title}\n본문: {body[:600]}\n\n후보:\n{listing}\n\n기호:"
    raw = _THINK_BLOCK.sub("", ollama_chat(model, PICK_SYSTEM, user, 0.0, 600, think=think) or "").strip()
    m = re.search(rf"[{LETTERS[: len(cands)]}]", raw.upper())
    return (LETTERS.index(m.group()) if m else 0), raw[:20]


class OfficialLineBuilder:
    def __init__(
        self,
        rules_path: Path = RULES,
        model: str | None = None,
        embed_model: str | None = None,
        k: int = 5,
        think: bool = False,
    ) -> None:
        self.cfg = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.org = OrgChart.load(Path(self.cfg["meta"]["org_chart"]))
        table_path = Path(self.cfg["meta"]["delegation_table"]).expanduser()
        embed = None
        if embed_model:
            from who_approved_this.tracks.t1d_draft import ollama_embed

            embed = ollama_embed(embed_model)
        self.lookup = DelegationLookup(
            load_table(table_path), self.cfg["execution"], embed=embed,
            vec_cache=table_path.with_name(f"delegation_vec_{embed_model.replace(':', '_')}.json")
            if embed_model else None,
        )
        self.model, self.k, self.think = model, k, think

    @property
    def org_name(self) -> str:
        return self.cfg["meta"]["org"]

    def depts(self) -> list[str]:
        return sorted(self.org.head)

    def decide_final(self, dept: str, title: str, body: str) -> dict[str, Any]:
        """종점(최소 전결 레벨)을 정한다. ``level`` 이 None 이면 판정 불가."""
        if self.lookup.is_execution(title):
            ex = self.lookup.execution(title, body)
            return {"source": ex["source"], "category": ex["category"], "amount": ex["amount"],
                    "amount_label": ex["amount_label"], "row": ex["row"], "level": ex["level"]}
        cands = self.lookup.candidates(dept, title + " " + body[:300], k=self.k)
        i, raw = pick_row(self.model, title, body, cands, self.think) if self.model else (0, None)
        return {
            "source": "llm_pick" if self.model else "top1", "category": None,
            "amount": None, "amount_label": None, "rid": cands[i]["rid"],
            "candidate_rids": [c["rid"] for c in cands], "llm_raw": raw,
            "row": f"[{cands[i]['sheet']}] {cands[i]['item']}", "level": cands[i]["final_level"],
            "candidates": [
                {"row": f"[{c['sheet']}] {c['item']}", "level": LEVEL_NAMES[c["final_level"]]}
                for c in cands
            ],
        }

    def team_lead(self, dept: str, team: str | None) -> str | None:
        """팀 이름("도시개발팀") 또는 팀장 직위("도시개발팀장")를 팀장 직위로."""
        if not team:
            return None
        if team.endswith(("팀장", "소장")):
            return team
        found = self.org.teams.get(dept, {}).get(team)
        return found["lead"] if found and found["lead"] else f"{team}장"

    def build(
        self,
        dept: str,
        drafter_title: str,
        team: str | None,
        title: str,
        body: str = "",
        date: str | None = None,
    ) -> dict[str, Any]:
        if dept not in self.org.head:
            return {"org": self.org_name, "dept": dept, "cells": [],
                    "needs_confirmation": [{"what": "부서", "why": f"조직도에 없는 부서: {dept}"}]}
        date = date or _date.today().isoformat()
        lead = self.team_lead(dept, team) if title_level(drafter_title) == 0 else None
        final = self.decide_final(dept, title, body)
        level = final["level"]
        cells = build_slot_chain(dept, drafter_title, lead, self.org, self.cfg,
                                 date=date, final_level=level if level is not None else 5)

        needs: list[dict[str, str]] = []
        if level is None:
            needs.append({"what": "금액",
                          "why": f"{final['category']} 집행 품의인데 금액을 못 찾음 — 구간 판정 불가, 군수까지 올림"})
        elif final["source"] != "amount_bracket":
            needs.append({"what": "전결 사무",
                          "why": f"별표 행을 검색·모델로 골랐다: {final['row'][:40]} → {LEVEL_NAMES[level]} "
                                 "(T6 행 판정 13/19). 후보를 확인할 것"})
        if level is not None and level < 5:
            needs.append({"what": "상향 결재",
                          "why": f"별표 최소 전결권자는 {LEVEL_NAMES[level]}. 더 높이 올릴지는 기안자가 정한다"})
        if lead and not self.org.team_of_lead(dept, lead):
            needs.append({"what": "팀", "why": f"조직도에 없는 팀장 직위: {lead}"})
        for c in cells:
            if "acting" in c["evidence"]:
                needs.append({"what": "직무대리",
                              "why": f"{c['slot']} 칸 → {c.get('acting_by') or c['title']} "
                                     "(rules/gapyeong.yaml acting, 인사 데이터로 확인할 것)"})
        return {
            "org": self.org_name, "dept": dept, "date": date,
            "final": final | {"title": LEVEL_NAMES.get(level) if level is not None else None},
            "cells": [{"role": c["role"], "title": c["title"], "slot": c["slot"],
                       "evidence": c["evidence"]} for c in cells],
            "needs_confirmation": needs,
        }


@lru_cache(maxsize=2)
def official_builder(model: str | None = None) -> OfficialLineBuilder:
    """프로세스당 하나. API·에이전트용."""
    return OfficialLineBuilder(model=model)
