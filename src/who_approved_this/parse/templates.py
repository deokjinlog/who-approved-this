"""초안 템플릿 로더와 렌더러.

템플릿은 ``templates/draft/*.yaml`` 에 유형별로 둔다. 각 칸에는 **값을 어디서 가져오는지**
(``source``)가 붙어 있고, 에이전트는 그 규칙대로만 채운다. 소스가 없으면 지어내지 않고
``[첨부 필요]`` · ``[연결문서 필요]`` 로 남긴다 — T1d 에서 자유 생성이 문서번호를 6/7
지어낸 걸 구조로 막는 장치다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: 소스가 없을 때 칸에 남기는 표시.
PLACEHOLDER = {
    "attachment": "[첨부 필요]",
    "linked_doc": "[연결문서 필요]",
    "input": "[입력 필요]",
    "generate": "[생성 실패]",
}


def load_draft_templates(directory: Path) -> dict[str, dict[str, Any]]:
    """``{work_type: template}``. ``"*"`` 가 기본 템플릿이다."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.yaml")):
        tpl = yaml.safe_load(path.read_text(encoding="utf-8"))
        tpl["_file"] = path.name
        out[tpl["work_type"]] = tpl
    return out


def pick_template(templates: dict[str, dict[str, Any]], work_type: str) -> dict[str, Any]:
    """유형 전용 템플릿이 있으면 그걸, 없으면 기본 템플릿을 돌려준다."""
    return templates.get(work_type) or templates["*"]


def render_markdown(title: str, sections: list[dict[str, Any]]) -> str:
    """채워진 칸을 공문 모양의 텍스트로 편다(사람이 읽는 형식)."""
    lines = [f"제목  {title}", ""]
    number = 1
    for sec in sections:
        value = (sec.get("value") or "").strip()
        style = sec.get("style", "item")
        if style == "paragraph":
            lines += [value, ""]
        elif style == "attachment":
            lines += ["", f"붙임  {value}  끝."]
        else:
            if sec.get("status") == "skipped":
                continue
            lines.append(f"{number}. {sec['label']}: {value}")
            number += 1
    return "\n".join(lines).strip() + "\n"
