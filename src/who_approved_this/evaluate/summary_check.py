"""요약 검증 — 환각률 · 실명 · 원문 위치 · 길이 비율.

사람 평가 없이 기계적으로 잴 수 있는 것만 잰다. 잘 썼는지(유용성)는
``DATA_ROOT/t1/summary_review/`` 의 (원문 / 요약) 쌍을 사람이 본다.

    환각률      생성 항목 속 숫자·날짜·금액 토큰 중 원문에 없는 비율.
                **필터 전(raw)** 으로 잰다 — 필터 후는 근거 검사를 통과한 것만 남아 0 이 당연하다.
    실명        출력에 명부 이름(자간 벌린 표기 포함)이 몇 번 나오나. 목표 0
    원문 위치   핵심 항목에 단 쪽 번호가 맞나. 그 항목의 낱말·숫자가 가장 많이 겹치는 쪽과 같으면 정답
    길이 비율   요약 글자 수 / 원문 글자 수
"""

from __future__ import annotations

import re
from typing import Any

_NUM = re.compile(r"\d[\d,.]*\d|\d")
_DATE = re.compile(r"(\d{4})\s*[.\-]\s*(\d{1,2})\s*[.\-]\s*(\d{1,2})")
_WORD = re.compile(r"[가-힣A-Za-z]{2,}")


def _flat(t: str) -> str:
    return re.sub(r"[\s,]", "", t)


def _tokens(item: str) -> list[str]:
    """숫자·날짜·금액 토큰. 날짜는 원문 표기와 비교하기 쉽게 숫자만 이어 붙인다."""
    toks = [f"{y}{int(m)}{int(d)}" for y, m, d in _DATE.findall(item)]
    rest = _DATE.sub(" ", item)
    toks += [_flat(n) for n in _NUM.findall(rest) if len(_flat(n)) >= 2]
    return toks


def hallucination(items: list[str], source: str) -> dict[str, Any]:
    src = _flat(source)
    src_dates = {f"{y}{int(m)}{int(d)}" for y, m, d in _DATE.findall(source)}
    total = bad = 0
    for it in items:
        for tok in _tokens(it):
            total += 1
            if tok not in src and tok not in src_dates:
                bad += 1
    return {"tokens": total, "ungrounded": bad, "rate": round(bad / total, 4) if total else 0.0}


def location_check(result: dict[str, Any], pages: list[str]) -> dict[str, Any]:
    """핵심 항목의 쪽 번호가 맞는지. 1쪽 문서는 채점하지 않는다."""
    if len(pages) < 2:
        return {"checked": 0, "correct": 0, "missing_page": 0}
    page_words = [set(_WORD.findall(p)) | set(_flat(n) for n in _NUM.findall(p)) for p in pages]
    checked = correct = missing = 0
    for kp in result["slots"]["locations"].get("key_points", []):
        if not kp.get("page"):
            missing += 1
            continue
        words = set(_WORD.findall(kp["value"])) | set(_tokens(kp["value"]))
        overlap = [len(words & pw) for pw in page_words]
        checked += 1
        correct += overlap[kp["page"] - 1] == max(overlap) and max(overlap) > 0
    return {"checked": checked, "correct": correct, "missing_page": missing}


def check(result: dict[str, Any], pages: list[str], known_names: set[str]) -> dict[str, Any]:
    source = "\n".join(pages)
    s = result["slots"]
    final_items = ([s["one_line"]] if s["one_line"] else []) + s["key_points"] + s["approver_checks"]
    out_text = " ".join(final_items + s["amounts"] + s["dates"] + s["doc_refs"] + s["vendors"])
    names = sum(1 for n in known_names if len(n) >= 2 and re.search(r"\s*".join(map(re.escape, n)), out_text))
    return {
        "hallucination_raw": hallucination(result.get("raw_items", []), source),
        "hallucination_final": hallucination(final_items, source),
        "names_in_output": names,
        "location": location_check(result, pages),
        "length_ratio": round(len(out_text) / max(len(source), 1), 4),
        "empty_slots": [k for k in ("one_line", "key_points", "amounts", "dates", "vendors", "approver_checks")
                        if not s.get(k)],
    }
