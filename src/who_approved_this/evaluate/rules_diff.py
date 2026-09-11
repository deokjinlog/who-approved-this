"""규칙 두 벌 대조 — 수기·엑셀(규정) vs 귀납.

미팅에서 쓸 문장은 이것이다 —
**"전결규정 없이 이력만으로 가면 어디서, 얼마나 틀리는가."**

세 가지를 잰다.
    * **커버리지** — 규정에는 있는데 이력에 안 나타난 규칙(표본이 못 본 조건).
    * **충돌** — 같은 조건인데 결론이 다른 규칙(이력이 잘못 일반화한 것).
    * **과잉 일반화** — 이력이 만든 규칙 중 규정에 근거가 없는 것.

비교 단위는 조직별 "항목"이다: 기안·결재 직위, 팀장 목록, 추가 경유, 협조 직위, 전결 금액 구간.
돌려주는 값에는 직위·조직 이름만 담고 사람 이름은 없다(규칙 dict 에 애초에 없다).
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _items(org: dict[str, Any]) -> dict[str, set[str]]:
    """조직 규칙을 비교 가능한 항목 집합으로 편다."""
    chain = org.get("chain", {}) or {}
    deleg = org.get("delegation", {}) or {}
    bands = {
        f"{r.get('work_type', '*')}|{r.get('amount_min')}~{r.get('amount_max')}→{r.get('final_title')}"
        for r in deleg.get("rules", []) or []
    }
    coop = set()
    for r in org.get("cooperation", []) or []:
        for t in r.get("titles", []):
            coop.add(t)
    return {
        "기안": {chain.get("draft", "")} - {""},
        "결재": {chain.get("approve", "") or deleg.get("default_final", "")} - {""},
        "팀장": {t["lead"] for t in org.get("teams", []) or []},
        "추가경유": {f"{x.get('work_type')}→{x.get('title')}" for x in org.get("extra_review", []) or []},
        "협조": coop,
        "전결구간": bands,
    }


def diff_rules(induced: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    """두 규칙 뭉치를 조직·항목별로 대조한다. ``official`` 은 수기 또는 엑셀 규칙."""
    a = {o["key"]: o for o in induced.get("organizations", [])}
    b = {o["key"]: o for o in official.get("organizations", [])}
    rows: list[dict[str, Any]] = []
    tot = Counter()
    for key in sorted(set(a) | set(b)):
        if key not in b:
            rows.append({"org": key, "item": "(조직)", "only_induced": ["규정에 없는 조직"], "only_official": [], "both": []})
            continue
        if key not in a:
            rows.append({"org": key, "item": "(조직)", "only_induced": [], "only_official": ["이력에 없는 조직"], "both": []})
            continue
        ia, ib = _items(a[key]), _items(b[key])
        for item in ia:
            both = ia[item] & ib[item]
            oi, oo = ia[item] - ib[item], ib[item] - ia[item]
            tot["official"] += len(ib[item])
            tot["covered"] += len(both)
            tot["only_induced"] += len(oi)
            # 기안·결재는 한 값이라 둘 다 있으면서 다르면 충돌
            if item in ("기안", "결재") and oi and oo:
                tot["conflict"] += 1
            rows.append({"org": key, "item": item, "both": sorted(both),
                         "only_induced": sorted(oi), "only_official": sorted(oo)})
    return {
        "rows": rows,
        "coverage_ratio": round(tot["covered"] / tot["official"], 4) if tot["official"] else None,
        "overgeneralized": tot["only_induced"],
        "conflict": tot["conflict"],
        "official_items": tot["official"],
    }


def diff_table(diff: dict[str, Any], only_changed: bool = True) -> str:
    """마크다운 표. 기본은 차이가 있는 행만."""
    lines = ["| 조직 | 항목 | 둘 다 | 귀납에만 | 규정에만 |", "|---|---|---|---|---|"]
    for r in diff["rows"]:
        if only_changed and not r["only_induced"] and not r["only_official"]:
            continue
        lines.append(f"| {r['org']} | {r['item']} | {', '.join(r['both']) or '-'} | "
                     f"{', '.join(r['only_induced']) or '-'} | {', '.join(r['only_official']) or '-'} |")
    lines.append("")
    lines.append(f"커버리지 {diff['coverage_ratio']} · 과잉 일반화 {diff['overgeneralized']} · 충돌 {diff['conflict']}"
                 f" (규정 항목 {diff['official_items']}개 기준)")
    return "\n".join(lines)


def score_predictions_by_source(
    cells: list[dict[str, Any]], evidence_field: str = "evidence"
) -> dict[str, Any]:
    """예측 칸을 **근거 태그별**로 갈라 정확도를 낸다.

    ``cells`` 는 ``{"title_src", "name_src", "title_ok", "name_ok"(None=채점 제외)}`` 목록.
    어느 층이 정확도를 올리고 어느 층이 틀리는지 분리해서 본다.
    """
    out: dict[str, Any] = {"title": {}, "name": {}}
    for kind in ("title", "name"):
        agg: dict[str, list[int]] = {}
        for c in cells:
            ok = c.get(f"{kind}_ok")
            if ok is None:
                continue
            agg.setdefault(c.get(f"{kind}_src", "?"), []).append(int(ok))
        out[kind] = {src: {"cells": len(v), "accuracy": round(sum(v) / len(v), 4)} for src, v in sorted(agg.items())}
    return out


def delegation_impact(diff: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """규정과 이력이 다른 조직에서 **실제로 틀린 칸 수**를 센다.

    ``cases`` 는 ``{"group"(조직), "cells": [{"title_ok", ...}]}`` 목록.
    """
    changed = {r["org"] for r in diff["rows"] if r["only_induced"] or r["only_official"]}
    wrong = Counter()
    for c in cases:
        if c.get("group") in changed:
            wrong[c["group"]] += sum(1 for x in c.get("cells", []) if x.get("title_ok") is False)
    return {"orgs_with_diff": sorted(changed), "wrong_title_cells": dict(wrong)}
