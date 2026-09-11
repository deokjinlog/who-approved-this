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
            + f'<td style="padding:4px 8px;color:#666">{(p or {}).get("evidence", "")}</td>'
            + "</tr>"
        )
    head = (
        "<tr>"
        + "".join(
            f'<th style="padding:4px 8px;text-align:left">{h}</th>'
            for h in ("#", "역할", "gold 직위", "gold 이름", "예측 직위", "예측 이름", "근거")
        )
        + "</tr>"
    )
    return (
        '<table style="border-collapse:collapse;font-size:14px">'
        + head
        + "".join(rows)
        + "</table>"
    )


def run_approval(choice: str, predictor: str = "default", rules: str = "induced") -> tuple[str, str]:
    """결재선 탭 실행 — 자기 자신을 참고에서 빼고(leave-one-out) 예측한다.

    본문은 결재란 이름을 가려서 준다(결재 전 문서엔 결재란이 없다).
    """
    doc_id = _doc_id(choice)
    if not doc_id:
        return "문서를 고르세요.", ""
    org, title = _org_of(doc_id), _title_of(doc_id)
    result = services.predict_approval_line(
        org, title, body=services.masked_body(doc_id), exclude_doc_id=doc_id, mask=False,
        predictor=predictor, rules=rules)
    gold = services.gold_cells(doc_id)
    summary = (
        f"**{doc_id}** · 조직 `{org}` · 예측기 `{result['predictor']}` · 규칙 `{result.get('rules') or '-'}` · "
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


def run_agent(choice: str, attach_text: str, linked: str, show_names: bool):
    """에이전트 탭 — 고른 문서의 제목과 기안자 직위로 새 문서를 기안하는 상황을 흉내 낸다."""
    doc_id = _doc_id(choice)
    if not doc_id:
        return "문서를 고르세요.", "", "", "", "", ""
    doc = next(d for d in services.list_docs() if d["doc_id"] == doc_id)
    drafter = next((c["title"] for c in services.gold_cells(doc_id) if c["role"] == "기안"), "주무관")
    request = {
        "org": doc["org"],
        "title_of_user": drafter,
        "title": doc.get("제목", ""),
        "exclude_doc_id": doc_id,
    }
    if attach_text.strip():
        request["attachments"] = [attach_text]
    if linked.strip():
        request["linked_docs"] = [x.strip() for x in linked.split(",") if x.strip()]
    r = services.agent_handle(request, mask=not show_names)

    wt = r.get("work_type", {})
    info = (
        f"**기안자** {r['user'].get('org')} · {r['user'].get('title')} ({r['user'].get('source')})  \n"
        f"**업무항목** {wt.get('value')} ({wt.get('source')})  \n"
        f"**시간** {r.get('timings')}"
    )
    rows = "".join(
        f"<tr><td style='padding:4px 8px'>{c['role']}</td><td style='padding:4px 8px'>{c['title']}</td>"
        f"<td style='padding:4px 8px'>{c.get('name') or '—'}</td>"
        f"<td style='padding:4px 8px;color:#666'>{c['evidence']}</td></tr>"
        for c in r["approval_line"]["cells"]
    )
    line_html = (
        "<table style='border-collapse:collapse;font-size:14px'><tr>"
        + "".join(f"<th style='padding:4px 8px;text-align:left'>{h}</th>" for h in ("역할", "직위", "이름", "근거"))
        + f"</tr>{rows}</table>"
    )
    sections = "\n".join(
        f"- **{s['label']}** `{s['source']}` → {s['status']}" for s in r["draft"]["sections"]
    ) + f"\n\n검사: {r['draft']['checks']}  \n참고 문서: {', '.join(r['draft']['references']) or '없음'}"
    summ = r.get("summary")
    summary_md = "(첨부·본문이 없어 요약하지 않음)" if not summ else (
        f"**한줄요약** {summ['slots']['one_line']}  \n"
        + "**핵심**  \n" + "\n".join(f"- {x}" for x in summ["slots"]["key_points"])
        + f"  \n**금액** {', '.join(summ['slots']['amounts']) or '—'}  \n"
        + f"**일정** {', '.join(summ['slots']['dates']) or '—'}  \n"
        + "**결재자 확인사항**  \n" + "\n".join(f"- {x}" for x in summ["slots"]["approver_checks"])
        + f"  \n검사: {summ['checks']}"
    )
    needs = "\n".join(f"- **{n['what']}** — {n['why']}" for n in r["needs_confirmation"]) or "없음"
    return info, line_html, r["draft"]["markdown"], sections, summary_md, needs


def run_summary(choice: str, text: str, mode: str) -> tuple[str, str]:
    """요약 탭 — 문서를 고르면 그 PDF 를 쪽별로, 아니면 붙여 넣은 본문을 요약한다."""
    doc_id = _doc_id(choice)
    if not doc_id and not text.strip():
        return "문서를 고르거나 본문을 붙여 넣으세요.", ""
    r = services.summarize_text(text=text or None, doc_id=doc_id or None, mode=mode)
    c = r["checks"]
    info = (f"모드 `{r['mode']}` · {r['pages']}쪽 · 모델 `{r['model'] or '(없음 — 추출 칸만)'}` · "
            f"근거 없는 항목 제외 {c['dropped_ungrounded_number'] + c['dropped_invented_ref']} · "
            f"출력 실명 {c['names_in_output']} · {c['sec']}s")
    return r["markdown"], info


def build() -> gr.Blocks:
    """화면 조립. 탭 네 개(결재선·초안·에이전트·요약)."""
    with gr.Blocks(title="who-approved-this 데모") as demo:
        gr.Markdown(
            "# who-approved-this — 로컬 데모\n"
            "공개 결재문서로 **결재선 예측**과 **품의서 초안 생성**을 확인한다. "
            "선택한 문서는 참고에서 제외한다(leave-one-out)."
        )

        with gr.Tab("결재선"):
            pick1 = gr.Dropdown(_doc_choices(), label="문서", value=None)
            with gr.Row():
                pred1 = gr.Dropdown(list(services.PREDICTORS), value="layered", label="예측기")
                rules1 = gr.Dropdown(services.rules_sets(), value="induced", label="규칙 벌 (layered)")
            go1 = gr.Button("실행", variant="primary")
            info1 = gr.Markdown()
            table1 = gr.HTML()
            gr.Markdown("근거 = `직위근거/이름근거` — roster(명부) · rule(전결·팀·협조 규칙) · vote(과거 다수결) · llm(후보 안 선택) · agent")
            go1.click(run_approval, inputs=[pick1, pred1, rules1], outputs=[info1, table1])

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

        with gr.Tab("에이전트"):
            gr.Markdown(
                "고른 문서의 **제목과 기안자 직위만** 넣고 새 문서를 기안하는 상황을 흉내 낸다. "
                "첨부 텍스트를 붙여 넣으면 초안의 첨부 칸과 요약이 채워진다."
            )
            pick3 = gr.Dropdown(_doc_choices(), label="문서", value=None)
            attach = gr.Textbox(label="첨부 텍스트 (선택)", lines=4)
            linked = gr.Textbox(label="관련 문서번호 (선택, 쉼표 구분)")
            show = gr.Checkbox(label="실명 표시 (로컬 확인용)", value=False)
            go3 = gr.Button("에이전트 실행", variant="primary")
            info3 = gr.Markdown()
            with gr.Row():
                line3 = gr.HTML(label="결재선")
                needs3 = gr.Markdown(label="확인 필요")
            with gr.Row():
                draft3 = gr.Textbox(label="초안", lines=18, max_lines=18)
                with gr.Column():
                    sections3 = gr.Markdown(label="초안 칸")
                    summary3 = gr.Markdown(label="요약")
            go3.click(
                run_agent,
                inputs=[pick3, attach, linked, show],
                outputs=[info3, line3, draft3, sections3, summary3, needs3],
            )

        with gr.Tab("요약"):
            gr.Markdown("본문 요약. 칸: 한줄요약 · 핵심 3가지 · 금액·일정·업체 · 결재자 확인사항 · 원문 위치(쪽). "
                        "원문에 없으면 비운다. **첨부 요약은 준비 중**(자리만).")
            pick4 = gr.Dropdown(_doc_choices(), label="문서 (또는 아래에 본문 붙여 넣기)", value=None)
            text4 = gr.Textbox(label="본문", lines=6)
            mode4 = gr.Radio(["whole", "paged"], value="whole", label="방식 (여러 쪽: 통째 / 쪽별 병합)")
            gr.File(label="첨부 (준비 중)", interactive=False)
            go4 = gr.Button("요약", variant="primary")
            out4 = gr.Markdown()
            info4 = gr.Markdown()
            go4.click(run_summary, inputs=[pick4, text4, mode4], outputs=[out4, info4])

    return demo


def main(host: str = "127.0.0.1", port: int = 7860) -> None:
    """로컬에서만 띄운다."""
    build().launch(server_name=host, server_port=port, share=False, inbrowser=False)
