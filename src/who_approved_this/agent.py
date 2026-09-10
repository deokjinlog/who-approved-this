"""전자결재 에이전트 — 단계가 고정된 워크플로형.

자유롭게 도구를 고르는 에이전트로 만들지 않았다. 지금까지 잰 숫자가 전부 반대를
가리켰기 때문이다.

    결재선    규칙 뼈대 95% > 이력 81% > LLM 단독 58%      → LLM 은 후보 안에서만 고른다
    초안      자유 생성은 문서번호 6/7 을 지어냈다          → 근거 없는 칸은 비워 둔다
    요약      근거를 주고 줄이면 숫자 근거율 99~100%        → 숫자는 정규식으로 옮긴다
    E2E       지표가 통과해도 화면에서 결함 6개             → 모든 출력에 근거를 붙인다

한 번 호출의 흐름::

    사용자 식별(명부)  →  업무항목(LLM)  →  결재선(전결표+조직도)
                      →  초안(템플릿 칸 채우기)  →  요약(첨부가 있으면)
                      →  확인 필요 목록

LLM 이 쓰이는 곳은 업무항목 분류, 초안 본문 문장, 요약 문장 셋뿐이다. 나머지는 규칙과
정규식이라 같은 입력에 같은 결과가 나오고, 틀리면 규칙 파일을 고치면 된다.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from who_approved_this.parse.approval_agent import build_line, load_rules
from who_approved_this.parse.summarizer import load_summary_template, summarize
from who_approved_this.parse.templates import (
    PLACEHOLDER,
    load_draft_templates,
    pick_template,
    render_markdown,
)
from who_approved_this.tracks.t1c_predictors import _THINK_BLOCK, ollama_chat
from who_approved_this.tracks.t1d_draft import NAME_MASK, mask_names
from who_approved_this.tracks.t1d_retrieval import BM25Retriever, chunk_documents, doc_type
from who_approved_this.tracks.t1e_worktype import classify_by_title
from who_approved_this.tracks.t1e_worktype import load_rules as load_worktypes

REPO = Path(__file__).resolve().parents[2]

#: 첨부 텍스트에서 초안 칸을 옮겨 올 때 쓰는 패턴. 공문 지출 서식의 항목명 기준.
ATTACH_FIELDS: dict[str, re.Pattern[str]] = {
    "amount": re.compile(r"(?:지\s*급\s*액|금\s*액|합\s*계)\s*[:：]?\s*([^\n]*\d[\d,]*\s*원[^\n]*)"),
    "breakdown": re.compile(r"산\s*출\s*내\s*역\s*[:：]?\s*([^\n]+)"),
    "period": re.compile(r"(?:사\s*용|근\s*무|사\s*업)?\s*기\s*간\s*[:：]\s*([^\n]+)"),
    "payee": re.compile(r"지\s*급\s*대\s*상\s*[:：]?\s*([^\n]+)"),
    "method": re.compile(r"결\s*제\s*방\s*법\s*[:：]?\s*([^\n]+)"),
    "budget": re.compile(r"예\s*산\s*과\s*목\s*[:：]?\s*([^\n]+)"),
}
_DOC_NO = re.compile(r"[가-힣]{2,12}-\d{2,7}")
_LAW = re.compile(r"[「『]([^」』]{2,40})[」』]")

DRAFT_SYSTEM = """너는 한국 행정기관의 공문서 작성자다. 주어진 칸만 채운다.

규칙:
- 참고 문서는 **형식·말투**만 참고한다. 내용을 옮기지 않는다.
- 금액·날짜·수량·문서번호·법령명을 쓰지 않는다. 그런 값은 다른 칸이 따로 채운다.
- 사람 이름을 쓰지 않는다. 참고 문서의 기관명·머리글을 베끼지 않는다.
- 작성할 문서의 제목이 다루는 사안만 쓴다.

출력은 JSON 객체 하나만. 키는 요청받은 칸의 key 이고 값은 그 칸의 문장이다."""

WT_SYSTEM = """너는 전자결재 시스템이다. 문서 제목과 본문을 보고 업무항목 하나를 고른다.
주어진 목록 안에서만 고른다. 출력은 항목 이름 한 줄만."""


def _json_object(raw: str) -> dict[str, Any] | None:
    from who_approved_this.parse.summarizer import _json_object as parse

    return parse(raw)


class ApprovalAgent:
    """전자결재 에이전트. ``handle()`` 하나로 결재선·초안·요약을 함께 낸다."""

    def __init__(self, model: str | None, docs: list[dict[str, Any]] | None = None) -> None:
        self.model = model
        self.rules = load_rules(REPO / "rules" / "approval.yaml")
        self.worktypes = load_worktypes()
        self.templates = load_draft_templates(REPO / "templates" / "draft")
        self.summary_template = load_summary_template(REPO / "templates" / "summary.yaml")
        self.docs = docs or []
        # 문서 로더마다 붙이는 필드가 달라(t1c 로더에는 doc_type 이 없다) 여기서 맞춘다.
        for d in self.docs:
            d.setdefault("doc_type", doc_type(d.get("제목", "")))
        self._people = self._load_people(Path(self.rules["roster"]["path"]).expanduser())
        self._retriever: BM25Retriever | None = None

    # --- 명부 -------------------------------------------------------------

    @staticmethod
    def _load_people(path: Path) -> list[dict[str, str]]:
        """명부 전체 행. **리포 밖 파일이고 실명이 들어 있다.**"""
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh, delimiter="\t"))

    @property
    def known_names(self) -> set[str]:
        return {p["name"] for p in self._people if p.get("name")}

    def _name_to_title(self, org: str) -> dict[str, str]:
        return {p["name"]: p["title"] for p in self._people if p["org"] == org}

    def _roster_for(self, org: str) -> dict[tuple[str, str], str]:
        return {(p["org"], p["title"]): p["name"] for p in self._people if p["org"] == org}

    def identify(
        self, user_name: str | None = None, org: str | None = None, title: str | None = None
    ) -> dict[str, Any]:
        """로그인 사용자를 조직·직위로 바꾼다. 결재선의 출발점이다."""
        if org and title:
            return {"ok": True, "org": org, "title": title, "source": "input"}
        hits = [p for p in self._people if user_name and p["name"] == user_name]
        if len(hits) == 1:
            return {"ok": True, "org": hits[0]["org"], "title": hits[0]["title"], "source": "roster"}
        if len(hits) > 1:
            return {"ok": False, "candidates": [(h["org"], h["title"]) for h in hits],
                    "reason": "명부에 같은 이름이 여러 명"}
        return {"ok": False, "reason": "명부에 없는 사용자"}

    # --- ① 업무항목 ---------------------------------------------------------

    def classify(self, title: str, body: str) -> tuple[str, str]:
        """업무항목을 고른다. 목록 밖 답이 오면 제목 규칙으로 물러난다."""
        types = self.worktypes["types"]
        valid = {t["key"] for t in types}
        if self.model:
            menu = "\n".join(f"- {t['key']}: {t['desc']}" for t in types)
            user = f"# 업무항목 목록\n\n{menu}\n\n# 제목\n\n{title}\n\n# 본문\n\n{body[:1500]}\n\n하나만 고르라."
            raw = ollama_chat(self.model, WT_SYSTEM, user, temperature=0.0, timeout=300) or ""
            lines = _THINK_BLOCK.sub("", raw).strip().splitlines()
            answer = lines[0].strip().strip('"') if lines else ""
            if answer in valid:
                return answer, "llm"
        return classify_by_title(title), "title-rule"

    # --- ③ 초안 ------------------------------------------------------------

    def _references(self, org: str, title: str, exclude: str | None) -> list[dict[str, Any]]:
        """같은 조직 기존 문서에서 형식 참고용 청크 3개. 조직 필터는 생성 오염을 막는다."""
        if self._retriever is None:
            self._retriever = BM25Retriever(chunk_documents(self.docs))
        hits = self._retriever.search(title, k=30)
        return [c for c, _ in hits if c["org"] == org and c["doc_id"] != exclude][:3]

    def draft(
        self,
        org: str,
        work_type: str,
        title: str,
        attachments: list[str] | None = None,
        linked_docs: list[str] | None = None,
        exclude_doc_id: str | None = None,
    ) -> dict[str, Any]:
        """템플릿의 칸을 소스 규칙대로 채운다. 소스가 없으면 지어내지 않는다."""
        template = pick_template(self.templates, work_type)
        attach_text = "\n".join(attachments or [])
        linked_docs = linked_docs or []
        refs = self._references(org, title, exclude_doc_id)
        names = self.known_names

        sections: list[dict[str, Any]] = []
        to_generate: list[dict[str, Any]] = []
        for spec in template["sections"]:
            sec = {k: spec[k] for k in ("key", "label", "source", "style") if k in spec}
            sec["required"] = bool(spec.get("required"))
            src = spec["source"]
            if src == "linked_doc":
                if linked_docs:
                    sec.update(value=", ".join(f"{d}호" for d in linked_docs) + "와 관련입니다.",
                               status="filled")
                else:
                    sec.update(value=PLACEHOLDER["linked_doc"], status="needs_input")
            elif src == "attachment":
                value = self._from_attachment(spec["key"], attach_text)
                if value:
                    sec.update(value=value, status="filled")
                else:
                    sec.update(value=PLACEHOLDER["attachment"], status="needs_input")
            elif src == "input":
                sec.update(value=title, status="filled")
            else:
                sec.update(value="", status="pending", hint=spec.get("hint", ""))
                to_generate.append(sec)
            sections.append(sec)

        checks = {"invented_refs": 0, "invented_laws": 0, "org_contamination": 0,
                  "names_masked": 0, "json_ok": not to_generate}
        if to_generate and self.model:
            generated = self._generate(title, to_generate, refs)
            checks["json_ok"] = generated is not None
            allowed = attach_text + " ".join(linked_docs)
            ref_text = " ".join(r["text"] for r in refs)
            for sec in to_generate:
                text = (generated or {}).get(sec["key"], "") or ""
                text, c = self._sanitize(text, org, allowed, ref_text, names)
                for k, v in c.items():
                    checks[k] += v
                sec.update(value=text or PLACEHOLDER["generate"],
                           status="generated" if text else "failed")
        elif to_generate:
            for sec in to_generate:
                sec.update(value=PLACEHOLDER["generate"], status="failed")

        return {
            "template": {"work_type": template["work_type"], "file": template["_file"],
                         "provenance": template.get("provenance")},
            "sections": sections,
            "markdown": render_markdown(title, sections),
            "references": [r["doc_id"] for r in refs],
            "checks": checks,
        }

    @staticmethod
    def _from_attachment(key: str, text: str) -> str:
        """첨부 텍스트에서 칸 값을 그대로 옮긴다. 없으면 빈 문자열."""
        if not text:
            return ""
        if key == "attachments":
            return ""
        pattern = ATTACH_FIELDS.get(key)
        if pattern:
            m = pattern.search(text)
            if m:
                return m.group(1).strip()[:120]
        if key == "amount":
            m = re.search(r"(\d{1,3}(?:,\d{3})+)\s*원", text)
            return f"금{m.group(1)}원" if m else ""
        return ""

    def _generate(
        self, title: str, sections: list[dict[str, Any]], refs: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """생성 칸들을 한 번에 요청한다. JSON 이 깨지면 한 번만 다시."""
        names = self.known_names
        spec = "\n".join(f'- "{s["key"]}": {s["label"]} — {s.get("hint", "")}' for s in sections)
        blocks = "\n\n".join(
            f"[형식 참고 {i} · {r['doc_type']}]\n{mask_names(r['text'][:800], names)}"
            for i, r in enumerate(refs, start=1)
        )
        user = (
            f"작성할 문서 제목: {title}\n\n# 채울 칸\n{spec}\n\n"
            f"# 같은 조직 기존 문서 (형식만 참고)\n\n{blocks or '(없음)'}\n\n"
            f"---\n제목: {title}\n위 칸들을 JSON 으로 채워라."
        )
        for _attempt in (1, 2):
            obj = _json_object(ollama_chat(self.model, DRAFT_SYSTEM, user, 0.3, 600) or "")
            if obj:
                return obj
        return None

    def _sanitize(
        self, text: str, org: str, allowed: str, ref_text: str, names: set[str]
    ) -> tuple[str, dict[str, int]]:
        """생성 문장에서 지어낸 문서번호·법령명·다른 기관명·실명을 걷어낸다."""
        counts = {"invented_refs": 0, "invented_laws": 0, "org_contamination": 0, "names_masked": 0}

        def _ref(m: re.Match[str]) -> str:
            if m.group(0) in allowed:
                return m.group(0)
            counts["invented_refs"] += 1
            return PLACEHOLDER["linked_doc"]

        text = _DOC_NO.sub(_ref, text)

        def _law(m: re.Match[str]) -> str:
            if m.group(1) in allowed or m.group(1) in ref_text:
                return m.group(0)
            counts["invented_laws"] += 1
            return "[근거 확인 필요]"

        text = _LAW.sub(_law, text)

        flat = re.sub(r"\s+", "", text)
        for other in {d["org"] for d in self.docs if d.get("org") and d["org"] != org}:
            if other in flat:
                counts["org_contamination"] += 1
        before = text
        text = mask_names(text, names)
        counts["names_masked"] = before.count(NAME_MASK) != text.count(NAME_MASK) and 1 or 0
        return text.strip(), counts

    # --- 한 번의 요청 -------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """요청 하나를 처리한다.

        ``request`` 키: ``user_name`` 또는 (``org``, ``title_of_user``), ``title``,
        ``body``(선택), ``attachments``(첨부 텍스트 목록, 선택),
        ``linked_docs``(관련 문서번호 목록, 선택), ``exclude_doc_id``(평가용).
        """
        timings: dict[str, float] = {}
        t = time.perf_counter()
        needs: list[dict[str, str]] = []

        user = self.identify(request.get("user_name"), request.get("org"),
                             request.get("title_of_user"))
        timings["identify"] = round(time.perf_counter() - t, 3)
        if not user["ok"]:
            return {"user": user, "needs_confirmation": [
                {"what": "기안자", "why": user.get("reason", "식별 실패")}], "timings": timings}
        org, drafter = user["org"], user["title"]

        title = request.get("title", "")
        body = request.get("body", "") or ""
        attachments = request.get("attachments") or []

        t = time.perf_counter()
        work_type, wt_source = self.classify(title, body)
        timings["classify"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        line = build_line(org, drafter, work_type, title, body + "\n" + "\n".join(attachments),
                          self.rules, self._roster_for(org))
        timings["approval_line"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        draft = self.draft(org, work_type, title, attachments, request.get("linked_docs"),
                           request.get("exclude_doc_id"))
        timings["draft"] = round(time.perf_counter() - t, 3)

        summary = None
        if attachments or body:
            t = time.perf_counter()
            pages = attachments or [body]
            summary = summarize(self.model, pages, self.summary_template,
                                self._name_to_title(org), self.known_names)
            timings["summary"] = round(time.perf_counter() - t, 3)

        # --- 확인 필요 목록: 에이전트가 모르는 것을 모른다고 말하는 곳 ---
        org_rules = next((o for o in self.rules["organizations"] if o["key"] == org), {})
        if not org_rules:
            needs.append({"what": "전결규정", "why": f"{org} 규칙 없음 — 결재선을 만들 수 없음"})
        elif org_rules.get("provenance") != "official":
            needs.append({"what": "최종 결재자",
                          "why": "전결규정 원문 미확인. 과거 결재 이력에서 귀납한 규칙이다"})
        for sec in draft["sections"]:
            if sec["status"] == "needs_input" and sec.get("required"):
                needs.append({"what": sec["label"], "why": sec["value"]})
        if summary and (summary["checks"]["dropped_ungrounded_number"]
                        or summary["checks"]["dropped_invented_ref"]):
            needs.append({"what": "요약", "why": "원문 근거가 없는 항목을 제외했다"})

        return {
            "user": user,
            "work_type": {"value": work_type, "source": wt_source},
            "approval_line": line,
            "draft": draft,
            "summary": summary,
            "needs_confirmation": needs,
            "timings": timings,
        }


def to_public(result: dict[str, Any], names: set[str]) -> dict[str, Any]:
    """API 로 내보낼 때 사람 이름을 가린다(결재선 칸의 name 과 본문 텍스트 전부)."""
    text = json.dumps(result, ensure_ascii=False)
    for name in sorted(names, key=len, reverse=True):
        if len(name) >= 2:
            text = re.sub(r"\s*".join(map(re.escape, name)), NAME_MASK, text)
    return json.loads(text)
