"""T1c 결재선 예측기 — 같은 인터페이스를 가진 세 가지 방식.

``predict(refs, target) -> [{"role", "title", "name"}]`` 하나만 맞추면 된다.

트랙 안에 두는 이유
    입력이 ``pdf_path`` 가 아니라 ``(참고 문서들, 대상 문서)`` 라서
    :class:`~who_approved_this.parse.base.Parser` 계약에 맞지 않는다.
    골격(base 인터페이스·패키지 구조)을 건드리지 않기로 했으므로,
    트랙 전용 모듈로 둔다.

방식
    ``baseline_copy``  같은 부서 직전 문서 결재선을 그대로 복사
    ``baseline_vote``  칸 위치별로 참고 문서 다수결(동률이면 최근 것)
    ``llm_local``      ollama 로컬 모델에 참고 문서와 대상 문서를 주고 JSON 요청
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from typing import Any, Protocol

#: 프롬프트에 넣을 본문 길이. 결재란 자체가 아니라 "무슨 문서인가"만 알면 된다.
BODY_CHARS = 1500

ROLES = ("기안", "검토", "결재", "협조")


class Predictor(Protocol):
    """결재선 예측기 공통 계약."""

    name: str

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        """참고 문서들을 보고 ``target`` 의 결재선을 예측한다."""
        ...


def _recent_first(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """생산일자 내림차순(같으면 문서번호 내림차순)으로 정렬한다."""
    return sorted(
        refs, key=lambda r: (r.get("생산일자", ""), r.get("문서번호", "")), reverse=True
    )


class BaselineCopy:
    """같은 부서 **직전 문서**의 결재선을 그대로 쓴다.

    "부서가 같으면 결재선도 같다"는 가장 단순한 가정. 이 가정이 어디서 깨지는지
    보는 게 목적이므로, 잘 맞든 안 맞든 그 자체가 결과다.
    """

    name = "baseline_copy"

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        if not refs:
            return []
        latest = _recent_first(refs)[0]
        return [dict(c) for c in latest["cells"]]


class BaselineVote:
    """칸 위치별로 참고 문서 **다수결**. 동률이면 최근 문서를 따른다."""

    name = "baseline_vote"

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        if not refs:
            return []
        ordered = _recent_first(refs)
        out: list[dict[str, str]] = []
        for role in ROLES:
            per_ref = [[c for c in r["cells"] if c["role"] == role] for r in ordered]
            n_cells = Counter(len(c) for c in per_ref).most_common(1)[0][0]
            for i in range(n_cells):
                cands = [cells[i] for cells in per_ref if i < len(cells)]
                if not cands:
                    continue
                out.append(
                    {
                        "role": role,
                        "title": _vote([c["title"] for c in cands]),
                        "name": _vote([c["name"] for c in cands]),
                    }
                )
        return out


def _vote(values: list[str]) -> str:
    """다수결. 동률이면 목록 앞쪽(=더 최근 문서)을 따른다."""
    counts = Counter(values)
    top = max(counts.values())
    for v in values:  # values 는 최근 순
        if counts[v] == top:
            return v
    return values[0]


class LocalLLM:
    """ollama 로컬 모델에 결재선을 물어본다.

    문서 내용(제목·본문)과 참고 결재선이 프롬프트에 들어가므로 **실명이 오간다.**
    ollama 는 로컬에서 돌고 외부로 나가지 않는다. 프롬프트 자체는 저장하지 않는다.
    """

    def __init__(self, model: str, timeout: int = 180) -> None:
        self.model = model
        self.timeout = timeout
        self.name = f"llm_local({model})"
        #: 마지막 호출의 원문 응답 — 파싱 실패 진단용. 저장하지 않는다.
        self.last_error = ""

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        prompt = build_prompt(refs, target)
        try:
            done = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.last_error = f"호출 실패: {type(exc).__name__}"
            return []
        return parse_cells(done.stdout, self)


SYSTEM = """너는 한국 행정기관의 전자결재 시스템이다. 문서의 결재선(기안·검토·결재·협조)을 예측한다.

규칙:
- 같은 부서의 과거 문서 결재선을 우선 근거로 삼는다.
- 결재선은 아래에서 위로 올라간다. 첫 칸이 기안자(실무자), 마지막 칸이 최종 결재자다.
- 직위(title)는 과거 문서에 나온 표기를 그대로 쓴다. 새 직위를 만들지 않는다.
- 이름(name)도 과거 문서에 나온 사람 중에서 고른다. 모르면 빈 문자열로 둔다.
- 문서 내용이 과거 문서와 다른 성격이면 칸 수나 직위가 달라질 수 있다.

출력은 JSON 배열 하나만. 설명·코드블록 없이 배열만 출력한다.
[{"role": "기안|검토|결재|협조", "title": "직위", "name": "이름"}]"""


def build_prompt(refs: list[dict[str, Any]], target: dict[str, Any]) -> str:
    """프롬프트를 만든다. 템플릿은 ``t1c_prompt.md`` 와 같은 구조다."""
    blocks = []
    for i, ref in enumerate(_recent_first(refs), start=1):
        line = "\n".join(
            f"{c['role']} | {c['title']} | {c['name']}" for c in ref["cells"]
        )
        blocks.append(
            f"## 과거문서 {i}\n제목: {ref['제목']}\n본문:\n{ref['body'][:BODY_CHARS]}\n"
            f"결재선:\n{line}"
        )
    return (
        f"{SYSTEM}\n\n"
        f"# 같은 부서의 과거 결재문서 {len(refs)}건\n\n"
        + "\n\n".join(blocks)
        + f"\n\n# 결재선을 예측할 문서\n\n제목: {target['제목']}\n본문:\n"
        f"{target['body'][:BODY_CHARS]}\n\n위 문서의 결재선을 JSON 배열로 출력하라."
    )


_JSON_ARRAY = re.compile(r"\[.*\]", re.S)


def parse_cells(raw: str, owner: LocalLLM | None = None) -> list[dict[str, str]]:
    """모델 출력에서 JSON 배열을 건져 칸 목록으로 만든다.

    설명이 앞뒤로 붙거나 코드블록으로 감싸도 배열만 뽑는다. 실패하면 빈 목록.
    """
    match = _JSON_ARRAY.search(raw)
    if not match:
        if owner is not None:
            owner.last_error = "JSON 배열 없음"
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        if owner is not None:
            owner.last_error = "JSON 파싱 실패"
        return []
    out: list[dict[str, str]] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        out.append(
            {
                "role": role if role in ROLES else "검토",
                "title": str(item.get("title", "")).strip(),
                "name": str(item.get("name", "")).strip(),
            }
        )
    return out
