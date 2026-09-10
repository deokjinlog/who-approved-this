"""T1e — 문서 → 전결표 업무항목 분류.

**결재선 자동 지정의 진짜 난제는 여기다.**

전결위임표는 "사무의 종류" 행마다 전결권자를 정해 둔다. 그러니 결재선을 만들려면
순서가 이렇다.

    ① 이 문서가 표의 **어느 행**인가        ← 어렵다. 이 트랙이 재는 것
    ② 표 조회 → 전결권자 직위               ← 단순 lookup, 100%
    ③ 조직도에서 기안자~전결권자 경로 생성   ← 결정론, 100%

②③은 표와 조직도만 있으면 코드로 끝난다. AI 가 필요한 건 ① 뿐이다.
전결표 행은 ``실·과소관 회계지출 및 물품관리`` 같은 **업무 분류**인데 문서 제목은
``'26년 8월분 초과근무수당 지출`` 이다. 그 사이를 잇는 게 ①이다.

측정 방법
    분류표와 정답 라벨은 ``rules/worktype.yaml``(사람이 붙임).
    LLM 에는 **제목을 주지 않고 본문만** 준다. 실제 기능에서는 기안 시점에 제목이
    아직 없거나 모호할 수 있고, 제목을 주면 제목에 답이 그대로 들어 있어
    (``…지출``, ``…계획``) 분류가 아니라 문자열 매칭이 되기 때문이다.

    비교군으로 **제목 키워드 규칙**도 함께 잰다. LLM 이 규칙을 넘어서는지 보려는 것이다.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.tracks.t1c_approval_predict import load_docs
from who_approved_this.tracks.t1c_predictors import _THINK_BLOCK, ollama_chat

TRACK = "t1e"

#: 분류표 위치(리포 안. 실명 없음).
RULES = Path(__file__).resolve().parents[3] / "rules" / "worktype.yaml"

#: 본문을 몇 자까지 보여줄지.
BODY_CHARS = 1800

#: 제목 키워드 규칙(비교군). 위에서부터 처음 걸리는 것.
TITLE_RULES = (
    ("회계지출", ("지출", "수당", "보험료", "렌탈료", "임금", "구입")),
    ("계약·검사", ("기성검사", "준공", "계약", "검사")),
    ("회의·심의", ("위원회", "협의회", "회의록", "심의")),
    ("인사·복무", ("임용", "호봉", "승진", "복무", "겸직")),
    ("사업계획", ("계획",)),
    ("대외제출·통보", ("제출", "통보", "승인", "알림")),
    ("결과보고", ("결과", "복명", "출장")),
)


def load_rules() -> dict[str, Any]:
    """분류표와 정답 라벨을 읽는다."""
    return yaml.safe_load(RULES.read_text(encoding="utf-8"))


def classify_by_title(title: str) -> str:
    """제목 키워드 규칙(비교군)."""
    flat = re.sub(r"\s+", "", title)
    for key, words in TITLE_RULES:
        if any(w in flat for w in words):
            return key
    return "결과보고"


def build_prompt(body: str, types: list[dict[str, str]]) -> str:
    """본문만 주고 업무항목을 고르게 한다. **제목은 주지 않는다.**"""
    menu = "\n".join(f"- {t['key']}: {t['desc']}" for t in types)
    return (
        f"# 업무항목 목록\n\n{menu}\n\n"
        f"# 문서 본문\n\n{body[:BODY_CHARS]}\n\n"
        "위 문서가 어느 업무항목인지 하나만 고르라."
    )


SYSTEM = """너는 한국 행정기관의 전자결재 시스템이다. 문서 본문을 읽고
전결위임표의 업무항목 중 하나로 분류한다.

규칙:
- 주어진 목록 안에서만 고른다. 새 항목을 만들지 않는다.
- 금액을 집행하는 문서는 회계지출이다.
- 무엇을 하겠다는 문서는 사업계획, 마치고 알리는 문서는 결과보고다.
- 위원회·협의회가 주체면 회의·심의다.

출력은 항목 이름 한 줄만. 설명을 붙이지 않는다."""


def run(model: str | None = None, max_docs: int | None = None) -> dict[str, Any]:
    """T1e를 돌린다. ``model`` 이 없으면 제목 규칙만 잰다."""
    rules = load_rules()
    types = rules["types"]
    labels = rules["labels"]
    valid = {t["key"] for t in types}

    docs = [d for d in load_docs(track_data_dir("t1")) if d.get("문서번호") in labels]
    results_dir = track_results_dir(TRACK)

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for d in docs:
        gold = labels[d["문서번호"]]
        by_title = classify_by_title(d["제목"])
        by_llm = ""
        sec = 0.0
        if model:
            t0 = time.perf_counter()
            raw = ollama_chat(
                model, SYSTEM, build_prompt(d["body"], types), temperature=0.0, timeout=300
            )
            sec = round(time.perf_counter() - t0, 2)
            text = _THINK_BLOCK.sub("", raw or "").strip().splitlines()
            answer = text[0].strip().strip('"').strip() if text else ""
            # 목록 밖 답은 버린다(후보 안에서만 고르게 하는 원칙).
            by_llm = answer if answer in valid else ""
        cases.append(
            {
                "doc_id": d["doc_id"],
                "문서번호": d["문서번호"],
                "gold": gold,
                "by_title": by_title,
                "by_llm": by_llm,
                "title_ok": by_title == gold,
                "llm_ok": by_llm == gold,
                "llm_off_menu": bool(model) and not by_llm,
                "elapsed_sec": sec,
            }
        )

    n = len(cases) or 1
    report = {
        "track": TRACK,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "문서 → 전결표 업무항목 분류 (결재선 자동 지정의 ① 단계)",
        "rules": str(RULES.relative_to(RULES.parents[2])),
        "types": [t["key"] for t in types],
        "model": model,
        "note": "LLM 에는 제목을 주지 않고 본문만 준다. 제목에 답이 들어 있기 때문이다.",
        "documents": len(cases),
        "summary": {
            "title_rule_accuracy": round(sum(c["title_ok"] for c in cases) / n, 4),
            "llm_accuracy": round(sum(c["llm_ok"] for c in cases) / n, 4) if model else None,
            "llm_off_menu": sum(c["llm_off_menu"] for c in cases),
            "sec_per_case": round(sum(c["elapsed_sec"] for c in cases) / n, 2),
        },
        "confusion": _confusion(cases),
        "cases": cases,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    out = results_dir / f"{datetime.now():%Y%m%d-%H%M}.json"
    if out.exists():
        out = results_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["results_path"] = str(out)
    return report


def _confusion(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """정답 항목별로 맞은 수와, 틀렸을 때 무엇으로 갔는지."""
    out: dict[str, Any] = {}
    for gold in sorted({c["gold"] for c in cases}):
        rows = [c for c in cases if c["gold"] == gold]
        out[gold] = {
            "n": len(rows),
            "title_ok": sum(r["title_ok"] for r in rows),
            "llm_ok": sum(r["llm_ok"] for r in rows),
            "llm_wrong_to": dict(
                Counter(r["by_llm"] or "(목록밖)" for r in rows if not r["llm_ok"])
            ),
        }
    return out


def report(rep: dict[str, Any]) -> None:
    """콘솔 출력. 문서 제목·본문은 찍지 않는다."""
    s = rep["summary"]
    typer.echo(f"[t1e] {rep['purpose']}")
    typer.echo(f"      업무항목 {len(rep['types'])}종 / 문서 {rep['documents']}건 / 모델 {rep['model'] or '(미지정)'}")
    typer.echo(f"      {rep['note']}")
    typer.echo("")
    typer.echo(f"  제목 키워드 규칙 : {s['title_rule_accuracy']:.1%}")
    if s["llm_accuracy"] is not None:
        typer.echo(
            f"  LLM (본문만)     : {s['llm_accuracy']:.1%}"
            f"   케이스당 {s['sec_per_case']}s"
            + (f"  목록밖 답 {s['llm_off_menu']}건" if s["llm_off_menu"] else "")
        )
    typer.echo("")
    typer.echo(f"  {'정답 업무항목':16} {'건수':>4} {'제목규칙':>8} {'LLM':>6}  틀린 답")
    for k, v in rep["confusion"].items():
        wrong = ", ".join(f"{a}×{b}" for a, b in v["llm_wrong_to"].items()) or "-"
        typer.echo(f"  {k:16} {v['n']:4} {v['title_ok']:8} {v['llm_ok']:6}  {wrong}")
    typer.echo("")
    typer.echo(f"  → {rep['results_path']}  ({rep['elapsed_sec']}s)")
