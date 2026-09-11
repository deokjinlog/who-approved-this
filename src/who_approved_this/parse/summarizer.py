"""템플릿 요약기 — 본문·첨부를 ``templates/summary.yaml`` 의 칸으로 요약한다.

칸은 성격이 셋이다.
    extract   원문에서 정규식으로 그대로 옮긴다(금액·날짜·문서번호). LLM 을 거치지 않는다.
    generate  LLM 이 쓴다(한줄요약·핵심·결재자 확인사항). 쓴 뒤 검증한다.
    system    코드가 채운다(원문 위치).

LLM 에 넘기기 **전에** 원문의 사람 이름을 직위로 바꾼다. 모델이 이름을 본 적이 없으면
출력에 이름을 쓸 수 없다. 넘긴 뒤에는 생성 항목마다 숫자·문서번호가 원문에 있는지
검사해, 근거 없는 항목은 버리고 센다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from who_approved_this.tracks.t1c_predictors import _THINK_BLOCK, ollama_chat

PATTERNS: dict[str, re.Pattern[str]] = {
    "amount": re.compile(r"(?<![\d,])(\d{1,3}(?:,\d{3})+|\d{4,})\s*원"),
    "date": re.compile(r"(\d{4})\s*[.\-]\s*(\d{1,2})\s*[.\-]\s*(\d{1,2})"),
    "doc_no": re.compile(r"[가-힣]{2,12}-\d{2,7}"),
}

#: 생성 항목의 숫자 토큰(2자리 이상). 원문에 없으면 근거 없음으로 본다.
_NUMBER = re.compile(r"\d[\d,.]*\d")

NAME_MASK = "○○○"

SYSTEM = """너는 전자결재 문서 요약 도우미다. 결재자가 30초 안에 문서를 파악하도록 돕는다.

규칙:
- 원문에 있는 내용만 쓴다. 원문에 없는 숫자·날짜·금액·문서번호·기관명·법령명을 만들지 않는다.
- 사람 이름은 쓰지 않는다. 필요하면 직위로 부른다.
- 결재자 확인사항에는 결재 전에 볼 점(금액·기간·근거 문서·빠진 첨부 등)을 짚는다.
- 원문으로 알 수 없으면 그 항목을 비운다.

출력은 JSON 객체 하나만. 설명이나 코드블록을 붙이지 않는다.
{"one_line": "한 문장", "key_points": ["...", "..."], "approver_checks": ["...", "..."]}"""


def load_summary_template(path: Path) -> dict[str, Any]:
    """``templates/summary.yaml`` 을 읽는다."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _flat(text: str) -> str:
    return re.sub(r"[\s,]", "", text)


def extract_all(pages: list[str]) -> dict[str, list[dict[str, Any]]]:
    """페이지별로 금액·날짜·문서번호를 뽑아 **원문 위치와 함께** 돌려준다."""
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in PATTERNS}
    seen: dict[str, set[str]] = {k: set() for k in PATTERNS}
    for page_no, page in enumerate(pages, start=1):
        for key, pattern in PATTERNS.items():
            for m in pattern.finditer(page):
                if key == "date":
                    y, mo, d = m.groups()
                    value = f"{y}-{int(mo):02d}-{int(d):02d}"
                elif key == "amount":
                    value = f"{m.group(1)}원"
                else:
                    value = m.group(0)
                if value not in seen[key]:
                    seen[key].add(value)
                    out[key].append({"value": value, "page": page_no})
    return out


def replace_names(text: str, name_to_title: dict[str, str]) -> tuple[str, int]:
    """원문의 사람 이름을 직위로 바꾼다. 직위를 모르면 ``○○○``.

    바로 앞에 이미 그 직위가 적혀 있으면("주무관 ○○○") 이름만 지운다 —
    "주무관 주무관" 이 되지 않게. 자간을 벌린 이름("홍 길 동")도 잡는다.
    """
    count = 0
    for name in sorted(name_to_title, key=len, reverse=True):
        if len(name) < 2:
            continue
        title = name_to_title.get(name) or ""
        pattern = re.compile(r"\s*".join(re.escape(ch) for ch in name))

        def _sub(m: re.Match[str]) -> str:
            before = text[max(0, m.start() - 12) : m.start()]
            if title and title in before:
                return ""
            return title or NAME_MASK

        text, n = pattern.subn(_sub, text)
        count += n
    return text, count


def _json_object(raw: str) -> dict[str, Any] | None:
    """출력에서 괄호 균형이 맞는 ``{...}`` 를 찾아 파싱한다."""
    text = _THINK_BLOCK.sub("", raw or "")
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                if isinstance(obj, dict):
                    return obj
    return None


def _grounded(item: str, source_flat: str) -> tuple[bool, str]:
    """생성 항목 하나가 원문에 근거가 있는지. 없으면 사유를 돌려준다."""
    for ref in PATTERNS["doc_no"].findall(item):
        if ref not in source_flat:
            return False, "invented_ref"
    for num in _NUMBER.findall(item):
        if _flat(num) not in source_flat:
            return False, "ungrounded_number"
    return True, ""


def summarize(
    model: str | None,
    pages: list[str],
    template: dict[str, Any],
    name_to_title: dict[str, str] | None = None,
    known_names: set[str] | None = None,
) -> dict[str, Any]:
    """페이지 목록을 받아 템플릿 칸을 채운다.

    ``name_to_title`` 은 원문 이름을 직위로 바꾸는 표(명부에서 온다),
    ``known_names`` 는 출력에 이름이 새지 않았는지 마지막에 검사할 명단이다.
    """
    name_to_title = name_to_title or {}
    source = "\n\n".join(pages)
    masked, names_replaced = replace_names(source, name_to_title)
    masked_pages = [replace_names(p, name_to_title)[0] for p in pages]
    found = extract_all(masked_pages)
    source_flat = _flat(masked)

    specs = {s["key"]: s for s in template["slots"]}
    slots: dict[str, Any] = {
        "amounts": [f["value"] for f in found["amount"]],
        "dates": [f["value"] for f in found["date"]],
        "doc_refs": [f["value"] for f in found["doc_no"]],
        "locations": {k: found[v] for k, v in (("amounts", "amount"), ("dates", "date"), ("doc_refs", "doc_no"))},
        "one_line": "",
        "key_points": [],
        "approver_checks": [],
    }
    checks = {
        "json_ok": False,
        "generated_items": 0,
        "dropped_ungrounded_number": 0,
        "dropped_invented_ref": 0,
        "names_replaced_in_source": names_replaced,
        "names_in_output": 0,
    }

    if model:
        user = f"# 원문\n\n{masked[:4000]}\n\n위 문서를 요약하라."
        obj = None
        for _attempt in (1, 2):
            obj = _json_object(ollama_chat(model, SYSTEM, user, temperature=0.1, timeout=300))
            if obj:
                break
        if obj:
            checks["json_ok"] = True
            for key in ("one_line", "key_points", "approver_checks"):
                spec = specs.get(key, {})
                value = obj.get(key, "" if key == "one_line" else [])
                items = [value] if isinstance(value, str) else [str(v) for v in value or []]
                kept: list[str] = []
                for item in items:
                    item = item.strip()[: spec.get("max_chars", 120)]
                    if not item:
                        continue
                    checks["generated_items"] += 1
                    ok, why = _grounded(item, source_flat)
                    if not ok:
                        checks[f"dropped_{why}"] += 1
                        continue
                    kept.append(item)
                kept = kept[: spec.get("max_items", len(kept) or 1)]
                slots[key] = (kept[0] if kept else "") if key == "one_line" else kept

    if known_names:
        out_text = json.dumps(slots, ensure_ascii=False)
        for name in known_names:
            if len(name) >= 2 and re.search(r"\s*".join(map(re.escape, name)), out_text):
                checks["names_in_output"] += 1

    return {"template": template.get("name", "summary"), "slots": slots, "checks": checks}


# ---------------------------------------------------------------------------
# v2 — 쪽 번호가 붙는 요약 (통째 / 쪽별 병합) + 업체 칸 + 마크다운
# ---------------------------------------------------------------------------

#: 업체 — "(주)○○", "○○(주)", "주식회사 ○○", "업체명: ○○", "계약상대자: ○○".
#: 요약 출력(리포 밖)에만 담는다. 리포 결과 JSON 에는 개수만(CLAUDE.md 예외 규칙).
_VENDOR = re.compile(
    r"(?:\(주\)|㈜|주식회사)\s*([가-힣A-Za-z0-9&]{2,20})"
    r"|([가-힣A-Za-z0-9&]{2,20})\s*(?:\(주\)|㈜)"
    r"|(?:업\s*체\s*명?|계약\s*상대자|상\s*호)\s*[:：]\s*([^\n,()]{2,30})"
)

_JOSA = re.compile(r"(?:와의|과의|으로|에게|와|과|이|가|은|는|을|를|의|에|로)$")

WHOLE_SYSTEM = SYSTEM.replace(
    '{"one_line": "한 문장", "key_points": ["...", "..."], "approver_checks": ["...", "..."]}',
    '{"one_line": "한 문장", "key_points": [{"text": "...", "page": 쪽번호}], "approver_checks": ["..."]}\n'
    "key_points 는 최대 3개, 각 항목에 그 내용이 나온 쪽 번호(원문의 '=== N쪽 ===' 표시)를 단다.",
)
PAGE_SYSTEM = """너는 전자결재 문서 요약 도우미다. 문서의 **한 쪽**만 받는다.
- 이 쪽에 있는 내용만 쓴다. 숫자·날짜·금액·문서번호를 만들지 않는다. 사람 이름은 쓰지 않는다.
- 요점이 없으면 빈 배열.
출력은 JSON 객체 하나만: {"points": ["...", "..."]}  (최대 3개)"""
MERGE_SYSTEM = """너는 전자결재 문서 요약 도우미다. 쪽별 요점 목록을 받아 문서 전체 요약을 만든다.
- 목록에 있는 내용만 쓴다. 숫자·날짜·금액·문서번호를 새로 만들지 않는다. 사람 이름은 쓰지 않는다.
- key_points 는 최대 3개, 각 항목에 근거가 된 요점의 쪽 번호를 그대로 단다.
출력은 JSON 객체 하나만:
{"one_line": "한 문장", "key_points": [{"text": "...", "page": 쪽번호}], "approver_checks": ["..."]}"""


def extract_vendors(pages: list[str]) -> list[dict[str, Any]]:
    out, seen = [], set()
    for page_no, page in enumerate(pages, start=1):
        for m in _VENDOR.finditer(page):
            value = next(g for g in m.groups() if g).strip()
            # "주식회사 ○○와" 처럼 조사가 붙어 잡힌다 — 끝의 조사를 뗀다(두 글자 이상 남을 때만)
            stripped = _JOSA.sub("", value)
            value = stripped if len(stripped) >= 2 else value
            if value and value not in seen:
                seen.add(value)
                out.append({"value": value, "page": page_no})
    return out


def _points(obj: dict[str, Any] | None) -> list[tuple[str, int | None]]:
    out = []
    for kp in (obj or {}).get("key_points", []) or []:
        if isinstance(kp, dict):
            page = kp.get("page")
            try:
                page = int(page) if page is not None else None
            except (TypeError, ValueError):
                page = None
            out.append((str(kp.get("text", "")).strip(), page))
        else:
            out.append((str(kp).strip(), None))
    return [(t, p) for t, p in out if t]


def summarize_doc(
    model: str | None,
    pages: list[str],
    template: dict[str, Any],
    name_to_title: dict[str, str] | None = None,
    known_names: set[str] | None = None,
    mode: str = "whole",
    num_ctx: int = 16384,
) -> dict[str, Any]:
    """v2 요약. ``mode`` = whole(쪽 구분 붙여 통째) | paged(쪽별 요약 → 병합).

    생성 항목은 **필터 전 원본(raw)도 함께** 돌려준다 — 환각률은 필터 전으로 잰다.
    필터 후는 근거 검사를 통과한 것만 남으니 0 이 나오는 게 당연하다.
    """
    import time as _time

    t0 = _time.perf_counter()
    name_to_title = name_to_title or {}
    masked_pages = [replace_names(p, name_to_title)[0] for p in pages]
    names_replaced = sum(replace_names(p, name_to_title)[1] for p in pages)
    found = extract_all(masked_pages)
    vendors = extract_vendors(masked_pages)
    source_flat = _flat("\n".join(masked_pages))
    specs = {s["key"]: s for s in template["slots"]}
    llm_calls = 0

    obj: dict[str, Any] | None = None
    if model:
        if mode == "paged" and len(masked_pages) > 1:
            lines = []
            for i, page in enumerate(masked_pages, start=1):
                raw = ollama_chat(model, PAGE_SYSTEM, f"# {i}쪽\n\n{page[:4000]}", 0.1, 300)
                llm_calls += 1
                for pt in (_json_object(raw) or {}).get("points", []) or []:
                    lines.append(f"[{i}쪽] {str(pt).strip()}")
            raw = ollama_chat(model, MERGE_SYSTEM, "# 쪽별 요점\n\n" + "\n".join(lines), 0.1, 300)
            llm_calls += 1
            obj = _json_object(raw)
        else:
            text = "\n\n".join(f"=== {i}쪽 ===\n{p}" for i, p in enumerate(masked_pages, start=1))
            for _attempt in (1, 2):
                raw = ollama_chat(model, WHOLE_SYSTEM, f"# 원문\n\n{text}\n\n위 문서를 요약하라.", 0.1, 600,
                                  num_ctx=num_ctx)
                llm_calls += 1
                obj = _json_object(raw)
                if obj:
                    break

    raw_items: list[str] = []
    one_line, key_points, pages_of, checks_ = "", [], [], []
    dropped = {"ungrounded_number": 0, "invented_ref": 0}
    if obj:
        cand_one = str(obj.get("one_line", "")).strip()[: specs["one_line"].get("max_chars", 80)]
        cand_pts = _points(obj)
        cand_chk = [str(x).strip() for x in obj.get("approver_checks", []) or [] if str(x).strip()]
        raw_items = ([cand_one] if cand_one else []) + [t for t, _ in cand_pts] + cand_chk

        def keep(item: str) -> bool:
            ok, why = _grounded(item, source_flat)
            if not ok:
                dropped[why] += 1
            return ok

        one_line = cand_one if cand_one and keep(cand_one) else ""
        for t, p in cand_pts:
            if keep(t[: specs["key_points"].get("max_chars", 90)]):
                key_points.append(t[: specs["key_points"].get("max_chars", 90)])
                pages_of.append(p if p and 1 <= p <= len(pages) else None)
        key_points, pages_of = key_points[:3], pages_of[:3]
        checks_ = [c[:90] for c in cand_chk if keep(c)][:3]

    slots = {
        "one_line": one_line,
        "key_points": key_points,
        "key_point_pages": pages_of,
        "amounts": [f["value"] for f in found["amount"]],
        "dates": [f["value"] for f in found["date"]],
        "vendors": [v["value"] for v in vendors],
        "doc_refs": [f["value"] for f in found["doc_no"]],
        "approver_checks": checks_,
        "locations": {"amounts": found["amount"], "dates": found["date"], "vendors": vendors,
                      "doc_refs": found["doc_no"],
                      "key_points": [{"value": k, "page": p} for k, p in zip(key_points, pages_of)]},
    }
    names_in_output = 0
    if known_names:
        out_text = json.dumps(slots, ensure_ascii=False)
        names_in_output = sum(1 for n in known_names
                              if len(n) >= 2 and re.search(r"\s*".join(map(re.escape, n)), out_text))
    return {
        "template": template.get("name", "summary"), "version": template.get("version", 1), "mode": mode,
        "slots": slots, "raw_items": raw_items,
        "checks": {"json_ok": obj is not None, "generated_items": len(raw_items),
                   "dropped_ungrounded_number": dropped["ungrounded_number"],
                   "dropped_invented_ref": dropped["invented_ref"],
                   "names_replaced_in_source": names_replaced, "names_in_output": names_in_output,
                   "llm_calls": llm_calls, "sec": round(_time.perf_counter() - t0, 2)},
    }


def render_markdown(result: dict[str, Any], title: str = "") -> str:
    """summary.md 모양으로 렌더링. 빈 칸은 '—' (채우지 않았다는 표시)."""
    s = result["slots"]

    def lst(values: list[str], locs: list[dict[str, Any]] | None = None) -> str:
        if not values:
            return "—"
        page = {x["value"]: x["page"] for x in (locs or [])}
        return ", ".join(f"{v} (p.{page[v]})" if v in page else v for v in values)

    pts = [f"{i}. {k}" + (f" (p.{p})" if p else "") for i, (k, p) in
           enumerate(zip(s["key_points"], s["key_point_pages"]), start=1)]
    lines = [f"# 요약{' — ' + title if title else ''}", "",
             f"**한줄요약** {s['one_line'] or '—'}", "",
             "**핵심 3가지**", *(pts or ["—"]), "",
             f"**금액** {lst(s['amounts'], s['locations']['amounts'])}",
             f"**일정** {lst(s['dates'], s['locations']['dates'])}",
             f"**업체** {lst(s['vendors'], s['locations']['vendors'])}",
             f"**관련 문서** {lst(s['doc_refs'], s['locations']['doc_refs'])}", "",
             "**결재자 확인사항**", *([f"- {c}" for c in s["approver_checks"]] or ["—"]), "",
             f"_모드 {result.get('mode')} · 근거 없는 항목 제외 "
             f"{result['checks']['dropped_ungrounded_number'] + result['checks']['dropped_invented_ref']}건_"]
    return "\n".join(lines) + "\n"
