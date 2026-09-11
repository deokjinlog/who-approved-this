"""layered 예측기 — 층을 위에서부터 쌓고, LLM 은 **후보 안에서만** 고른다.

    ① roster          (조직, 직위) → 이름 후보. 1명이면 확정
    ② 전결·대결        최종 결재 직위, 전결 칸이면 이름 비움
    ③ doc_type        업무항목·금액
    ④ 팀라우팅          첫 검토 칸 = 주제 담당 팀장
    ⑤ 협조규칙          협조 칸
    ⑥ 남는 칸 vote      규칙이 안 정한 검토 칸
    ⑦ LLM             팀 동점·이름 후보 여럿일 때만, 후보 목록 안에서 기호로 고른다

칸마다 ``evidence = "직위근거/이름근거"`` (roster · rule · vote · llm).

규칙 벌
    ``induced``  케이스마다 **참고 문서만으로** 귀납(대상 문서가 규칙에 새지 않게)
    ``manual``   ``rules/approval.yaml`` (사람이 gold 전체를 보고 쓴 것 — 대상 문서가 섞여 있다)
    ``<path>``   엑셀 등에서 만든 규칙 파일. 조직 키로 찾는다
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from who_approved_this.parse.approval_layers import (
    build_roster,
    classify_doc,
    induce_org,
    lookup_cooperation,
    lookup_delegation,
    lookup_extra_review,
    lookup_roster,
    recent_first,
    route_team,
)
from who_approved_this.tracks.t1c_predictors import _THINK_BLOCK, BaselineVote, ollama_chat

REPO = Path(__file__).resolve().parents[3]
LETTERS = "ABCDEFGHIJKLMNOP"
PICK_SYSTEM = "너는 전자결재 시스템이다. 주어진 후보 중 하나만 기호(A, B, …)로 답한다. 설명하지 않는다."


def load_rules_set(rules: str) -> dict[str, dict[str, Any]] | None:
    """규칙 벌 이름 → ``{조직 키: 조직 규칙}``. ``induced`` 는 None(케이스마다 귀납)."""
    if rules == "induced":
        return None
    path = REPO / "rules" / "approval.yaml" if rules == "manual" else Path(rules).expanduser()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {o["key"]: o for o in data.get("organizations", [])}


class LayeredPredictor:
    def __init__(self, model: str | None = None, rules: str = "induced", timeout: int = 300) -> None:
        self.model = model
        self.rules_name = rules
        self.rules_set = load_rules_set(rules)
        self.timeout = timeout
        self.name = f"layered[{Path(rules).stem if rules not in ('induced', 'manual') else rules}]" + (
            f"({model})" if model else "")
        self.llm_calls = 0
        self.retries = self.failures = 0

    # --- LLM 은 후보 안에서만 ---------------------------------------------------
    def _pick(self, question: str, options: list[str], context: str) -> int | None:
        if not self.model or len(options) < 2:
            return None
        listing = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
        raw = ollama_chat(self.model, PICK_SYSTEM, f"{context}\n\n{question}\n{listing}\n\n기호:",
                          0.0, self.timeout)
        self.llm_calls += 1
        m = re.search(rf"[{LETTERS[: len(options)]}]", _THINK_BLOCK.sub("", raw or "").upper())
        return LETTERS.index(m.group()) if m else None

    def predict(self, refs: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, str]]:
        org = target["org"]
        if self.rules_set is None:
            org_rules = induce_org(refs, org) if refs else {}
        else:
            org_rules = self.rules_set.get(org, {})
        skeleton = BaselineVote().predict(refs, target)
        if not org_rules:  # 규칙 없는 조직 — 전부 vote
            return [{**c, "evidence": "vote/vote"} for c in skeleton]

        title, body = target["제목"], target["body"]
        context = f"문서 제목: {title}\n본문(앞부분, 이름 가림): {body[:800]}"
        info = classify_doc(title, body)                                         # ③
        deleg = lookup_delegation(org_rules, info["work_type"], info["amount"])   # ②
        roster = build_roster(refs)                                               # ①
        cells: list[dict[str, str]] = []

        def add(role: str, title_: str, src: str) -> None:
            cells.append({"role": role, "title": title_, "name": "", "t_src": src})

        vote_by_role: dict[str, list[str]] = {}
        for c in skeleton:
            vote_by_role.setdefault(c["role"], []).append(c["title"])

        # 기안
        draft = org_rules.get("chain", {}).get("draft")
        add("기안", draft or (vote_by_role.get("기안") or [""])[0], "rule" if draft else "vote")

        # 검토 — ④ 팀라우팅 → 추가 경유 → ⑥ 남는 칸 vote
        team = route_team(org_rules, title, body)
        leads = [t["lead"] for t in org_rules.get("teams", []) or []]
        reviews: list[tuple[str, str]] = []
        if leads:
            if team["lead"]:
                reviews.append((team["lead"], "rule"))
            else:
                ctx_teams = [
                    f"{t['lead']} — 담당 주제: {', '.join(t.get('keywords', [])[:6]) or '(키워드 없음)'}"
                    for t in org_rules["teams"]
                ]
                i = self._pick("이 문서를 검토할 팀장은?", ctx_teams, context)
                reviews.append((leads[i], "llm") if i is not None else (team["prior"], "vote"))
        for t in lookup_extra_review(org_rules, info["work_type"]):
            if t != deleg["final_title"] and all(t != r for r, _ in reviews):
                reviews.append((t, "rule"))
        voted = [t for t in vote_by_role.get("검토", []) if t not in leads]
        want = max(org_rules.get("chain", {}).get("review_slots", len(vote_by_role.get("검토", []))), len(reviews))
        for t in voted:
            if len(reviews) >= want:
                break
            if t != deleg["final_title"]:
                reviews.append((t, "vote"))
        for t, src in reviews:
            add("검토", t, src)

        # 결재 — ②
        add("결재", deleg["final_title"], "rule")

        # 협조 — ⑤. 규칙 벌에 협조 항목이 없으면 vote 가 남긴 협조를 쓴다
        if "cooperation" in org_rules:
            for c in lookup_cooperation(org_rules, info["work_type"], title, body):
                add("협조", c["title"], "rule")
        else:
            for t in vote_by_role.get("협조", []):
                add("협조", t, "vote")

        # 이름 — ① roster, 여럿이면 ⑦ LLM, 전결 칸이면 비움
        past = recent_first(refs)
        name_empty = bool(deleg.get("delegated_cell", {}).get("name_empty"))
        for c in cells:
            if c["role"] == "결재" and name_empty:
                c["name"], n_src = "", "rule"
            else:
                cands = lookup_roster(roster, org, c["title"])
                if len(cands) == 1:
                    c["name"], n_src = cands[0], "roster"
                elif len(cands) > 1:
                    opts = []
                    for nm in cands:
                        seen = [d["제목"][:30] for d in past
                                if any(x["name"] == nm and x["title"] == c["title"] for x in d["cells"])][:2]
                        opts.append(f"{nm} — 과거 담당: {' / '.join(seen)}")
                    i = self._pick(f"'{c['title']}' 칸에 들어갈 사람은?", opts, context)
                    c["name"], n_src = (cands[i], "llm") if i is not None else (cands[0], "vote")
                else:
                    n_src = "none"
            c["evidence"] = f"{c.pop('t_src')}/{n_src}"
        return cells
