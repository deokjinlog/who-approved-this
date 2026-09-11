"""툴콜링 에이전트 예측기 — layered 의 층을 도구로 주고, **호출 순서는 모델이 정한다.**

layered 와 같은 층 함수(:mod:`~who_approved_this.parse.approval_layers`)를 래핑만 한다.
규칙·명부는 layered[induced] 와 똑같이 케이스마다 참고 문서만으로 만든다 —
두 예측기의 차이는 "순서를 코드가 정하나, 모델이 정하나" 하나뿐이다.

    도구 6개   classify_doc · lookup_delegation · route_team · lookup_cooperation ·
               lookup_roster · search_past
    시스템     도구 설명 + "후보 밖 이름 금지" + 출력 형식. 순서는 지시하지 않는다
    제한       도구 호출 최대 8회. 넘으면 실패(빈 결재선)

관찰 대상은 모델이 **어떤 순서로** 부르는지다. 케이스마다 호출 이름 순서만 남긴다(인자·결과 값은
이름이 섞일 수 있어 저장하지 않는다).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from who_approved_this.parse.approval_layers import (
    build_roster,
    classify_doc,
    induce_org,
    lookup_cooperation,
    lookup_delegation,
    lookup_roster,
    route_team,
    search_past,
)
from who_approved_this.tracks.t1c_predictors import parse_cells

MAX_CALLS = 8

SYSTEM = """너는 전자결재 시스템이다. 도구로 정보를 모아 주어진 문서의 결재선을 만든다.
- 사람 이름은 도구가 돌려준 후보 안에서만 쓴다. 후보 밖 이름은 금지다. 모르면 빈 문자열로 둔다.
- 마지막 답은 JSON 배열 하나만 출력한다:
[{"role": "기안|검토|결재|협조", "title": "직위", "name": "이름"}]"""


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_S = {"type": "string"}
TOOLS = [
    _fn("classify_doc", "문서 제목·본문으로 업무항목(회계지출·계약·사업계획·결과보고·인사·회의·대외제출)과 금액을 판정한다.",
        {"title": _S, "body": _S}, ["title"]),
    _fn("lookup_delegation", "전결 규칙을 조회해 최종 결재 직위를 돌려준다. 업무항목과 금액(원)이 조건이다.",
        {"work_type": _S, "amount": {"type": "integer"}}, ["work_type"]),
    _fn("route_team", "사업 주제로 검토할 담당 팀장을 고른다. 점수와 동점 여부도 돌려준다.",
        {"title": _S, "body": _S}, ["title"]),
    _fn("lookup_cooperation", "업무항목과 문서 주제로 협조 칸(협조 직위)을 조회한다.",
        {"work_type": _S, "title": _S, "body": _S}, ["work_type", "title"]),
    _fn("lookup_roster", "직위에 해당하는 사람 이름 후보를 명부에서 찾는다.",
        {"title": _S}, ["title"]),
    _fn("search_past", "같은 조직 과거 문서 중 제목이 비슷한 것의 결재선을 찾는다.",
        {"query": _S, "k": {"type": "integer"}}, ["query"]),
]


def _chat(model: str, messages: list[dict[str, Any]], timeout: int) -> dict[str, Any] | None:
    body = {"model": model, "messages": messages, "tools": TOOLS, "stream": False,
            "think": False, "options": {"temperature": 0.0}}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None


def _int(v: Any) -> int | None:
    try:
        return int(str(v).replace(",", "").replace("원", "").strip())
    except (TypeError, ValueError):
        return None


class AgentPredictor:
    def __init__(self, model: str, max_calls: int = MAX_CALLS, timeout: int = 300) -> None:
        self.model = model
        self.max_calls = max_calls
        self.timeout = timeout
        self.name = f"agent({model})"
        self.retries = 0
        self.failures = 0
        #: 마지막 케이스의 호출 기록 — 도구 이름 순서와 판정만. 값은 담지 않는다.
        self.last_trace: dict[str, Any] = {}

    def predict(self, refs: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, str]]:
        org = target["org"]
        org_rules = induce_org(refs, org) if refs else {}
        roster = build_roster(refs)
        title, body = target["제목"], target["body"]
        known = {n for names in roster.values() for n in names}

        tools = {
            "classify_doc": lambda a: classify_doc(a.get("title") or title, a.get("body") or body),
            "lookup_delegation": lambda a: lookup_delegation(org_rules, str(a.get("work_type", "")), _int(a.get("amount"))),
            "route_team": lambda a: route_team(org_rules, a.get("title") or title, a.get("body") or body),
            "lookup_cooperation": lambda a: lookup_cooperation(
                org_rules, str(a.get("work_type", "")), a.get("title") or title, a.get("body") or body),
            "lookup_roster": lambda a: {"title": a.get("title", ""),
                                        "candidates": lookup_roster(roster, org, str(a.get("title", "")))},
            "search_past": lambda a: search_past(refs, str(a.get("query") or title), _int(a.get("k")) or 3),
        }
        trace: dict[str, Any] = {"calls": [], "failed": None, "unknown_tool": 0, "bad_args": 0,
                                 "out_of_candidate": 0, "rounds": 0}
        self.last_trace = trace
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"조직: {org}\n제목: {title}\n본문(이름 가림):\n{body[:1500]}"},
        ]
        content = ""
        t0 = time.perf_counter()
        while True:
            trace["rounds"] += 1
            resp = _chat(self.model, messages, self.timeout)
            if resp is None:
                trace["failed"] = "ollama 호출 실패"
                break
            msg = resp.get("message", {})
            calls = msg.get("tool_calls") or []
            if not calls:
                content = msg.get("content", "")
                break
            if len(trace["calls"]) + len(calls) > self.max_calls:
                trace["calls"].extend(c.get("function", {}).get("name", "?") for c in calls)
                trace["failed"] = f"도구 호출 {self.max_calls}회 초과"
                break
            messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
            for c in calls:
                fn = c.get("function", {}).get("name", "?")
                args = c.get("function", {}).get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args, trace["bad_args"] = {}, trace["bad_args"] + 1
                trace["calls"].append(fn)
                if fn not in tools:
                    trace["unknown_tool"] += 1
                    result: Any = {"error": f"모르는 도구: {fn}"}
                else:
                    try:
                        result = tools[fn](args if isinstance(args, dict) else {})
                    except Exception as e:  # noqa: BLE001 — 도구 오류도 관찰 대상이다
                        trace["bad_args"] += 1
                        result = {"error": type(e).__name__}
                messages.append({"role": "tool", "tool_name": fn,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
        trace["sec"] = round(time.perf_counter() - t0, 2)

        if trace["failed"]:
            self.failures += 1
            return []
        cells = parse_cells(content)
        if not cells:
            trace["failed"] = "최종 JSON 없음"
            self.failures += 1
            return []
        for c in cells:
            if c["name"] and c["name"] not in known:
                trace["out_of_candidate"] += 1
            c["evidence"] = "agent/agent"
        return cells
