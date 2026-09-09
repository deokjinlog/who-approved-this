"""Gradio 데모 화면 — 결재선 예측과 초안 생성을 눈으로 본다.

로컬 데모다. 화면에는 실명이 그대로 나온다(비교하려면 필요하다).
**스크린샷을 리포에 넣지 않는다.** 계산은 전부 :mod:`who_approved_this.api.services`
를 호출한다 — 화면에서 로직을 다시 쓰지 않는다.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from who_approved_this.api import services

#: 틀린 칸에 입힐 배경색.
BAD = "#ffd9d9"
GOOD = "#e8f5e9"


def _doc_choices() -> list[str]:
    """드롭다운 항목: ``문서id (조직 / 유형)``."""
    return [
        f"{d['doc_id']}  ({d['org']} / {d['doc_type']})" for d in services.list_docs()
    ]


def _doc_id(choice: str) -> str:
    return choice.split("  (")[0] if choice else ""


def _org_of(doc_id: str) -> str:
    doc = services.get_doc(doc_id)
    return doc["org"] if doc else ""


def _title_of(doc_id: str) -> str:
    doc = next((d for d in services.list_docs() if d["doc_id"] == doc_id), None)
    return doc.get("제목", "") if doc else ""


def _cells_table(gold: list[dict[str, str]], pred: list[dict[str, str]]) -> str:
    """gold 와 예측을 칸 단위로 나란히 놓고, 틀린 칸에 빨간 배경을 준다."""
    rows = []
    for i in range(max(len(gold), len(pred))):
        g = gold[i] if i < len(gold) else None
        p = pred[i] if i < len(pred) else None
        title_ok = bool(g and p and g["title"] == p["title"])
        name_ok = bool(g and p and g["name"] == p["name"])
        cell = lambda v, ok: (
            f'<td style="background:{GOOD if ok else BAD};padding:4px 8px">{v or "—"}</td>'
        )
        rows.append(
            "<tr>"
            f'<td style="padding:4px 8px">{i + 1}</td>'
            f'<td style="padding:4px 8px">{(g or {}).get("role", "—")}</td>'
            f'<td style="padding:4px 8px">{(g or {}).get("title", "—")}</td>'
            f'<td style="padding:4px 8px">{(g or {}).get("name", "—")}</td>'
            + cell((p or {}).get("title"), title_ok)
            + cell((p or {}).get("name"), name_ok)
            + "</tr>"
        )
    head = (
        "<tr>"
        + "".join(
            f'<th style="padding:4px 8px;text-align:left">{h}</th>'
            for h in ("#", "역할", "gold 직위", "gold 이름", "예측 직위", "예측 이름")
        )
        + "</tr>"
    )
    return (
        '<table style="border-collapse:collapse;font-size:14px">'
        + head
        + "".join(rows)
        + "</table>"
    )


def run_approval(choice: str) -> tuple[str, str]:
    """결재선 탭 실행 — 자기 자신을 참고에서 빼고(leave-one-out) 예측한다."""
    doc_id = _doc_id(choice)
    if not doc_id:
        return "문서를 고르세요.", ""
    org, title = _org_of(doc_id), _title_of(doc_id)
    result = services.predict_approval_line(org, title, exclude_doc_id=doc_id)
    gold = services.gold_cells(doc_id)
    summary = (
        f"**{doc_id}** · 조직 `{org}` · 예측기 `{result['predictor']}` · "
        f"참고 문서 {result['reference_count']}건\n\n"
        f"참고: {', '.join(result['reference_doc_ids']) or '(없음)'}"
    )
    return summary, _cells_table(gold, result["cells"])


def run_draft(choice: str) -> tuple[str, str, str, str, str, str]:
    """초안 탭 실행 — 제목만으로 검색해 초안을 만든다."""
    doc_id = _doc_id(choice)
    if not doc_id:
        return "문서를 고르세요.", "", "", "", "", ""
    org, title = _org_of(doc_id), _title_of(doc_id)
    result = services.generate_draft(org, title, n=3, exclude_doc_id=doc_id)
    refs = "\n".join(
        f"- `{r['doc_id']}` ({r['org']} / {r['doc_type']}) — 점수 {r['score']}"
        for r in result["references"]
    )
    header = (
        f"**{doc_id}** · 제목: {title}\n\n"
        f"검색 `{result['retriever']}` · 생성 `{result['model'] or '(미지정)'}`"
        + (f"\n\n> {result['note']}" if result.get("note") else "")
    )
    drafts = list(result["drafts"]) + [""] * 3
    return header, services.actual_body(doc_id), drafts[0], drafts[1], drafts[2], refs


def build() -> gr.Blocks:
    """화면 조립. 탭 두 개."""
    with gr.Blocks(title="who-approved-this 데모") as demo:
        gr.Markdown(
            "# who-approved-this — 로컬 데모\n"
            "공개 결재문서로 **결재선 예측**과 **품의서 초안 생성**을 확인한다. "
            "선택한 문서는 참고에서 제외한다(leave-one-out)."
        )

        with gr.Tab("결재선"):
            pick1 = gr.Dropdown(_doc_choices(), label="문서", value=None)
            go1 = gr.Button("실행", variant="primary")
            info1 = gr.Markdown()
            table1 = gr.HTML()
            go1.click(run_approval, inputs=pick1, outputs=[info1, table1])

        with gr.Tab("초안"):
            pick2 = gr.Dropdown(_doc_choices(), label="문서", value=None)
            go2 = gr.Button("실행", variant="primary")
            info2 = gr.Markdown()
            with gr.Row():
                actual = gr.Textbox(label="실제 문서", lines=22, max_lines=22)
                with gr.Column():
                    with gr.Tab("초안 1"):
                        d1 = gr.Textbox(label="", lines=20, max_lines=20)
                    with gr.Tab("초안 2"):
                        d2 = gr.Textbox(label="", lines=20, max_lines=20)
                    with gr.Tab("초안 3"):
                        d3 = gr.Textbox(label="", lines=20, max_lines=20)
                refs = gr.Markdown(label="검색된 참고 문서")
            go2.click(run_draft, inputs=pick2, outputs=[info2, actual, d1, d2, d3, refs])

    return demo


def main(host: str = "127.0.0.1", port: int = 7860) -> None:
    """로컬에서만 띄운다."""
    build().launch(server_name=host, server_port=port, share=False, inbrowser=False)
