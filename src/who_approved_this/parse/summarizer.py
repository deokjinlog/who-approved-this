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
