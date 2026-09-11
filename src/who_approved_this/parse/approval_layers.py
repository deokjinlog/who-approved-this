"""결재선 층(layer) 함수와 gold 귀납 — layered 예측기와 툴콜링 에이전트가 **같은 함수**를 쓴다.

층 (위에서부터)
    classify_doc        제목·본문 → 업무항목·금액                     (규칙)
    lookup_delegation   업무항목·금액 → 최종 결재 직위 + 전결 칸 정책   (규칙, priority 첫 매칭 → 폴백)
    route_team          사업 주제 → 담당 팀장                          (규칙, 키워드)
    lookup_cooperation  업무항목·주제 → 협조 칸                        (규칙)
    lookup_roster       (조직, 직위) → 이름 후보                       (명부)
    search_past         같은 조직 과거 문서의 결재선                   (이력)

규칙 dict 의 모양은 ``rules/approval.yaml`` (schema v2) 과 같다. 수기·귀납·엑셀 어느 벌이든
같은 함수로 조회한다. 조직 키나 시트 이름은 이 파일에 없다 — 전부 규칙 dict 에서 온다.

**이름은 규칙 dict 에 넣지 않는다.** 명부(roster)는 따로 받는다(리포 밖 데이터).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from who_approved_this.parse.approval_agent import extract_amount
from who_approved_this.tracks.t1e_worktype import classify_by_title

EMPTY_TITLES = {"", "-"}
_WORD = re.compile(r"[가-힣A-Za-z0-9]{2,}")
#: 주제 키워드에서 뺄 말 — 어느 문서에나 붙는 행정 상투어.
STOPWORDS = {
    "관련", "계획", "결과", "보고", "운영", "추진", "알림", "요청", "실시", "대한", "위한",
    "개최", "제출", "검토", "사항", "관한", "따른", "안내", "협조", "시행", "방안", "현황",
}


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------

def cells_of(doc: dict[str, Any], role: str) -> list[dict[str, str]]:
    return [c for c in doc["cells"] if c["role"] == role and c["title"] not in EMPTY_TITLES]


def recent_first(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(docs, key=lambda d: (d.get("생산일자", ""), d.get("문서번호", "")), reverse=True)


def _mode(values: list[str]) -> str:
    """최빈값. 동률이면 앞(=최근)."""
    if not values:
        return ""
    counts = Counter(values)
    top = max(counts.values())
    return next(v for v in values if counts[v] == top)


#: 직함 **앞뒤**의 2~4자 낱말은 사람 이름일 수 있다("국회의원 ○○○", "○○○ 의원"). 키워드에서 뺀다.
_TITLES_OF_PERSON = r"(?:국회의원|의원|위원장|위원|장관|교수|대표|님|씨)"
_HONORIFIC = re.compile(rf"{_TITLES_OF_PERSON}[\s(（]*([가-힣]{{2,4}})")
_HONORIFIC_BEFORE = re.compile(rf"([가-힣]{{2,4}})[\s)）]*{_TITLES_OF_PERSON}")


def person_like(text: str) -> set[str]:
    """제목 속 사람 이름 후보(직함 앞뒤). 직함 자체와 상투어는 뺀다."""
    found = set(_HONORIFIC.findall(text)) | set(_HONORIFIC_BEFORE.findall(text))
    return {w for w in found if not re.fullmatch(_TITLES_OF_PERSON, w) and w not in STOPWORDS}


def title_words(text: str) -> set[str]:
    """주제 키워드 후보. 상투어·숫자 섞인 말(7월, 1차)·**직함 앞뒤 사람 이름**은 뺀다."""
    persons = person_like(text)
    return {w for w in _WORD.findall(text)
            if w not in STOPWORDS and w not in persons and not any(ch.isdigit() for ch in w)}


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


# ---------------------------------------------------------------------------
# 층 함수 — 에이전트 도구가 이 시그니처 그대로 래핑한다
# ---------------------------------------------------------------------------

def classify_doc(title: str, body: str) -> dict[str, Any]:
    """문서 성격 판정. 업무항목은 제목 규칙, 금액은 정규식."""
    return {"work_type": classify_by_title(title), "amount": extract_amount(body)}


def lookup_delegation(org_rules: dict[str, Any], work_type: str, amount: int | None) -> dict[str, Any]:
    """전결 조회. ``priority`` 오름차순 첫 매칭 1건, 없으면 ``fallback``.

    반환: ``{"final_title", "matched", "note", "delegated_cell"}``.
    ``delegated_cell.name_empty`` 가 참이면 전결로 결재가 생략된 칸 — 직위는 찍히고 이름은 비운다.
    """
    deleg = org_rules.get("delegation", {}) or {}
    rules = sorted(deleg.get("rules", []) or [], key=lambda r: r.get("priority", 1))
    for rule in rules:
        if rule.get("work_type", "*") not in ("*", work_type) and work_type not in _as_list(rule.get("work_type")):
            continue
        lo, hi = rule.get("amount_min"), rule.get("amount_max")
        if lo is not None and (amount is None or amount < lo):
            continue
        if hi is not None and (amount is None or amount > hi):
            continue
        return {"final_title": rule["final_title"], "matched": True,
                "note": rule.get("note", "전결표"), "delegated_cell": deleg.get("cell_policy", {})}
    fb = deleg.get("fallback") or {}
    final = fb.get("final_title") or deleg.get("default_final") or org_rules.get("chain", {}).get("approve", "")
    return {"final_title": final, "matched": False, "note": f"폴백:{fb.get('policy', 'default_final')}",
            "delegated_cell": deleg.get("cell_policy", {})}


def route_team(org_rules: dict[str, Any], title: str, body: str) -> dict[str, Any]:
    """사업 주제로 담당 팀장을 고른다. 제목 적중 3점, 본문 1점.

    반환: ``{"lead", "scores", "ambiguous", "prior"}`` — 1위가 0점이거나 동점이면 ambiguous.
    ``prior`` 는 관측 문서가 가장 많은 팀(동점·0점일 때 규칙이 기댈 곳).
    """
    teams = org_rules.get("teams", []) or []
    if not teams:
        return {"lead": None, "scores": {}, "ambiguous": False, "prior": None}
    t_flat = re.sub(r"\s+", "", title)
    b_flat = re.sub(r"\s+", "", body[:1500])
    scores = {}
    for team in teams:
        s = 0
        for kw in team.get("keywords", []):
            s += 3 if kw in t_flat else (1 if kw in b_flat else 0)
        scores[team["lead"]] = s
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_s = ranked[0]
    ambiguous = best_s == 0 or (len(ranked) > 1 and ranked[1][1] == best_s)
    prior = max(teams, key=lambda t: t.get("observed_docs", 0))["lead"]
    return {"lead": None if ambiguous else best, "scores": scores, "ambiguous": ambiguous, "prior": prior}


def lookup_extra_review(org_rules: dict[str, Any], work_type: str) -> list[str]:
    """업무항목별 추가 경유(예: 지출은 예산 담당 팀장을 한 번 더)."""
    return [x["title"] for x in org_rules.get("extra_review", []) or []
            if x.get("work_type") in ("*", work_type) or work_type in _as_list(x.get("work_type"))]


def lookup_cooperation(org_rules: dict[str, Any], work_type: str, title: str, body: str) -> list[dict[str, str]]:
    """협조 칸. ``required`` 거나, 업무항목이 맞고 trigger 키워드가 문서에 있을 때."""
    hay = re.sub(r"\s+", "", title + body[:1500])
    out: list[dict[str, str]] = []
    for rule in org_rules.get("cooperation", []) or []:
        wts = _as_list(rule.get("work_type")) or ["*"]
        if "*" not in wts and work_type not in wts:
            continue
        words = rule.get("trigger_keywords", []) or []
        hit = rule.get("required", False) or (bool(words) and any(w in hay for w in words))
        if not hit:
            continue
        why = "required" if rule.get("required") else "keyword:" + next(w for w in words if w in hay)
        for t in rule.get("titles", []):
            if all(o["title"] != t for o in out):
                out.append({"title": t, "mode": rule.get("mode", "parallel"), "why": why})
    return out


def lookup_roster(roster: dict[tuple[str, str], list[str]], org: str, title: str) -> list[str]:
    """(조직, 직위) → 이름 후보(최근 순). 명부는 리포 밖 데이터."""
    return list(roster.get((org, title), []))


def search_past(refs: list[dict[str, Any]], query: str, k: int = 3) -> list[dict[str, Any]]:
    """같은 조직 과거 문서 중 제목이 비슷한 것 k건의 결재선."""
    q = title_words(query)
    scored = sorted(recent_first(refs), key=lambda d: -len(q & title_words(d["제목"])))
    return [
        {"doc_id": d["doc_id"], "제목": d["제목"], "work_type": classify_by_title(d["제목"]),
         "cells": [{"role": c["role"], "title": c["title"], "name": c["name"]}
                   for c in d["cells"] if c["title"] not in EMPTY_TITLES]}
        for d in scored[:k]
    ]


def build_roster(docs: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    """문서들의 결재선에서 명부를 만든다. 이름은 최근 문서 순, 중복 제거."""
    roster: dict[tuple[str, str], list[str]] = {}
    for d in recent_first(docs):
        for c in d["cells"]:
            name = c.get("name", "")
            if c["title"] in EMPTY_TITLES or name in ("", "-", "?"):
                continue
            names = roster.setdefault((d["org"], c["title"]), [])
            if name not in names:
                names.append(name)
    return roster


# ---------------------------------------------------------------------------
# 귀납 — gold 결재선에서 같은 스키마의 규칙을 만든다
# ---------------------------------------------------------------------------

def induce_org(docs: list[dict[str, Any]], org: str) -> dict[str, Any]:
    """한 조직의 문서들로 규칙을 귀납한다. **이름은 담지 않는다.**

    * chain       기안·결재 직위의 최빈값
    * teams       첫 검토 칸이 문서마다 갈리면 → 그 직위들을 팀으로, 제목 키워드로 라우팅
    * extra_review 검토 칸이 평소보다 많은 문서의 추가 직위 + 그 문서의 업무항목
    * delegation  최종 결재 최빈값(priority 1, 전 구간) + 폴백. 관측 금액 범위는 note
    * cooperation 협조 칸이 있는 문서의 직위 + 업무항목 + 협조 없는 문서엔 없는 제목 키워드
    * acting      "대결" 표기가 관측된 칸 (직무대행 — 부재는 예측 불가라 기록만)
    """
    docs = recent_first(docs)
    n = len(docs)
    draft = _mode([cells_of(d, "기안")[0]["title"] for d in docs if cells_of(d, "기안")])
    approve = _mode([cells_of(d, "결재")[-1]["title"] for d in docs if cells_of(d, "결재")])

    reviews = [[c["title"] for c in cells_of(d, "검토")] for d in docs]
    first = [r[0] for r in reviews if r]
    teams: list[dict[str, Any]] = []
    if len(set(first)) >= 2:
        by_team: dict[str, list[dict[str, Any]]] = {}
        for d, r in zip(docs, reviews):
            if r:
                by_team.setdefault(r[0], []).append(d)
        words = {t: Counter(w for d in ds for w in title_words(d["제목"])) for t, ds in by_team.items()}
        for t, ds in by_team.items():
            others = set().union(*(set(words[o]) for o in by_team if o != t))
            kws = [w for w, _ in words[t].most_common() if w not in others][:12]
            teams.append({"name": t[:-1] if t.endswith("팀장") else t, "lead": t, "keywords": kws,
                          "observed_docs": len(ds), "provenance": "induced"})

    modal_len = Counter(len(r) for r in reviews).most_common(1)[0][0] if reviews else 0
    extra: Counter[tuple[str, str]] = Counter()
    for d, r in zip(docs, reviews):
        if len(r) > modal_len:
            for t in r[modal_len:]:
                extra[(classify_by_title(d["제목"]), t)] += 1
    extra_review = [{"work_type": wt, "title": t, "observed_docs": k, "provenance": "induced"}
                    for (wt, t), k in extra.items()]

    amounts = [a for d in docs if (a := extract_amount(d["body"])) is not None]
    finals = [cells_of(d, "결재")[-1] for d in docs if cells_of(d, "결재")]
    empty_final = sum(1 for c in finals if c.get("name", "") in ("", "-"))

    acting = []
    for d in docs:
        for c in d["cells"]:
            if "대결" in c.get("note", ""):
                acting.append({"slot": approve, "by": c["title"], "observed_docs": 1,
                               "provenance": "induced", "note": "대결 표기 관측 — 부재 여부는 문서만으로 예측 불가"})

    coop_docs = [d for d in docs if cells_of(d, "협조")]
    plain_words = set().union(*(title_words(d["제목"]) for d in docs if not cells_of(d, "협조")))
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for d in coop_docs:
        titles = tuple(c["title"] for c in cells_of(d, "협조"))
        rule = merged.setdefault(titles, {"titles": list(titles), "work_type": [], "trigger_keywords": [],
                                          "mode": "parallel", "mode_provenance": "assumed",
                                          "observed_docs": 0, "provenance": "induced"})
        wt = classify_by_title(d["제목"])
        if wt not in rule["work_type"]:
            rule["work_type"].append(wt)
        for w in sorted(title_words(d["제목"]) - plain_words):
            if w not in rule["trigger_keywords"]:
                rule["trigger_keywords"].append(w)
        rule["observed_docs"] += 1

    return {
        "key": org,
        "provenance": "induced",
        "observed_docs": n,
        "join_key": "org+title",  # 전결사항표 ↔ 결재계층을 잇는 키: (조직, 직위 표준 명칭)
        "chain": {"draft": draft, "approve": approve, "review_slots": modal_len},
        "teams": teams,
        "delegation": {
            "default_final": approve,
            "rules": [{"priority": 1, "work_type": "*", "amount_min": None, "amount_max": None,
                       "final_title": approve, "provenance": "induced",
                       "note": (f"관측 {len(finals)}건, 금액 {min(amounts):,}~{max(amounts):,}원"
                                if amounts else f"관측 {len(finals)}건, 금액 없음")}],
            "fallback": {"policy": "chain_top", "final_title": approve,
                         "note": "규칙 미매칭 시 최상위 결재 직위 — 상신을 막지도, 무결재로 통과시키지도 않는다"},
            "cell_policy": {"name_empty": bool(finals) and empty_final * 2 > len(finals),
                            "observed_name_empty": empty_final},
        },
        "acting": acting,
        "extra_review": extra_review,
        "cooperation": list(merged.values()),
    }


def induce_rules(docs: list[dict[str, Any]], generated_at: str = "") -> dict[str, Any]:
    """문서 전체 → 조직별 귀납 규칙 (``rules/approval.induced.yaml`` 모양)."""
    by_org: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        by_org.setdefault(d["org"], []).append(d)
    return {
        "meta": {"schema_version": 3, "source": "induced", "generated_at": generated_at,
                 "source_docs": len(docs),
                 "note": "gold 결재선에서 자동 귀납. 이름 없음. priority·fallback·acting·mode·join_key 는 "
                         "조사(approval_rules_survey.md §8-A)에서 나온 빠진 개념의 자리"},
        "organizations": [induce_org(ds, org) for org, ds in sorted(by_org.items())],
    }
