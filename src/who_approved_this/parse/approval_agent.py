"""결재선 자동 지정 — 기안자와 문서를 받아 결재선을 만든다.

**T1b 와 반대 방향이다.** T1b 는 문서에 찍힌 결재선을 *읽어냈고*, 여기는 아직 없는
결재선을 *만든다*. 회사 기능 ②(결재라인 자동 삽입)가 이쪽이다.

실제 기능과 맞춘 전제
    기안자는 **입력으로 주어진다**(로그인한 사용자). 예측하지 않는다.
    T1c 에서 기안자 이름까지 맞히려 한 건 인위적인 난이도였다.

네 단계, 그중 하나만 AI
    ① 업무항목 분류      LLM. 본문을 읽고 전결표의 어느 행인지 정한다
    ② 금액 추출          정규식
    ③ 전결권자 조회      전결표 lookup. 결정론
    ④ 검토 경로 생성     조직도 + 담당업무. 결정론

    ②③④는 표와 조직도만 정확하면 100% 다. AI 가 필요한 건 ① 뿐이고,
    그래서 규칙 파일(``rules/approval.yaml``)의 품질이 전체 정확도를 좌우한다.

칸마다 근거를 단다
    ``evidence`` 에 ``rule`` / ``team`` / ``roster`` / ``llm`` 을 남긴다.
    사용자가 고칠 때 왜 그렇게 정해졌는지 보이고, 수정 로그가 다음 규칙이 된다.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import yaml

#: 금액. "금10,474,360원" 형태를 우선 잡고, 없으면 콤마 낀 큰 수.
_AMOUNT = re.compile(r"금\s*([\d,]{4,})\s*원")
_AMOUNT_LOOSE = re.compile(r"([\d]{1,3}(?:,[\d]{3}){1,})\s*원")


def load_rules(path: Path) -> dict[str, Any]:
    """``rules/approval.yaml`` 을 읽는다."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_roster(path: Path) -> dict[tuple[str, str], str]:
    """``roster.tsv`` 를 ``{(조직, 직위): 이름}`` 으로. **리포 밖 파일이다.**"""
    if not path.expanduser().is_file():
        return {}
    with path.expanduser().open(encoding="utf-8") as fh:
        return {
            (r["org"], r["title"]): r["name"]
            for r in csv.DictReader(fh, delimiter="\t")
        }


def extract_amount(body: str) -> int | None:
    """본문에서 금액을 뽑는다. 전결 구간 판정에 쓴다."""
    for pattern in (_AMOUNT, _AMOUNT_LOOSE):
        m = pattern.search(body)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def pick_team(org_rules: dict[str, Any], title: str, body: str) -> dict[str, Any] | None:
    """사업 주제로 담당 팀을 고른다.

    **전결표가 아니라 조직도 축이다.** 전결표는 최종 결재자를 정하고, 어느 팀이
    검토하는지는 담당업무가 정한다. gold 에서 "독서 코칭"은 독서문화진흥팀,
    "스튜디오·AI"는 운영팀으로 갈렸는데 둘 다 업무항목은 같았다.

    제목에 가중치를 둔다 — 제목이 사업 주제를 가장 직접적으로 담는다.
    """
    best, best_score = None, 0
    haystack_title = re.sub(r"\s+", "", title)
    haystack_body = re.sub(r"\s+", "", body[:1500])
    for team in org_rules.get("teams", []):
        score = 0
        for kw in team.get("keywords", []):
            if kw in haystack_title:
                score += 3
            elif kw in haystack_body:
                score += 1
        if score > best_score:
            best, best_score = team, score
    return best


def final_title(org_rules: dict[str, Any], work_type: str, amount: int | None) -> tuple[str, str]:
    """전결표를 조회해 최종 결재 직위와 근거를 돌려준다."""
    for rule in org_rules.get("delegation", {}).get("rules", []):
        wt = rule.get("work_type", "*")
        if wt not in ("*", work_type):
            continue
        lo, hi = rule.get("amount_min"), rule.get("amount_max")
        if lo is not None and (amount is None or amount < lo):
            continue
        if hi is not None and (amount is None or amount > hi):
            continue
        return rule["final_title"], rule.get("note", "전결표")
    return org_rules.get("delegation", {}).get("default_final", ""), "기본값"


def build_line(
    org: str,
    drafter_title: str,
    work_type: str,
    title: str,
    body: str,
    rules: dict[str, Any],
    roster: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    """결재선을 만든다.

    ``drafter_title`` 은 **입력**이다(로그인 사용자의 직위). 예측하지 않는다.
    """
    roster = roster or {}
    org_rules = next(
        (o for o in rules.get("organizations", []) if o["key"] == org), None
    )
    if org_rules is None:
        return {"org": org, "cells": [], "error": f"규칙에 없는 조직: {org}"}

    amount = extract_amount(body)
    team = pick_team(org_rules, title, body)
    final, final_note = final_title(org_rules, work_type, amount)

    def cell(role: str, cell_title: str, evidence: str) -> dict[str, str]:
        return {
            "role": role,
            "title": cell_title,
            "name": roster.get((org, cell_title), ""),
            "evidence": evidence,
        }

    cells = [cell("기안", drafter_title, "input")]

    # ④ 검토 경로 — 담당 팀장을 먼저, 그다음 업무항목별 추가 경유
    if team and team["lead"] != final:
        cells.append(cell("검토", team["lead"], f"team:{team['name']}"))
    for extra in org_rules.get("extra_review", []):
        if extra.get("work_type") in ("*", work_type):
            t = extra["title"]
            if t != final and all(c["title"] != t for c in cells):
                cells.append(cell("검토", t, f"rule:{extra.get('note', '추가경유')[:20]}"))

    cells.append(cell("결재", final, f"delegation:{final_note[:24]}"))

    for coop in org_rules.get("cooperation", []):
        if coop.get("work_type") not in ("*", work_type):
            continue
        # required 면 무조건, 아니면 trigger_keywords 가 문서에 있을 때만 붙인다.
        # 협조는 문서 성격이 정한다 — 같은 회계지출이라도 인건비는 경리 협조가 붙고
        # 물품 구입은 안 붙는다(gold 12012 vs 10185).
        hit = coop.get("required", False)
        if not hit:
            words = coop.get("trigger_keywords", [])
            hay = re.sub(r"\s+", "", title + body[:1500])
            hit = bool(words) and any(w in hay for w in words)
        if hit:
            for t in coop.get("titles", []):
                cells.append(cell("협조", t, "rule:cooperation"))

    return {
        "org": org,
        "work_type": work_type,
        "amount": amount,
        "team": team["name"] if team else None,
        "final_title": final,
        "cells": cells,
    }
