"""typer CLI 진입점.

    uv run wat run t4
    uv run wat run t4 --pages 3 --docs 1   # 첫 문서 앞 3페이지만
"""

from __future__ import annotations

import typer

from who_approved_this.tracks import (
    t1_approval_line,
    t1c_approval_predict,
    t1d_draft,
    t5_gazette_summary,
    t4_gazette_ocr,
)

app = typer.Typer(
    help="who-approved-this — 공개 공공데이터 문서 파싱 벤치마크",
    no_args_is_help=True,
)

#: 트랙 이름 → 실행 함수. 트랙이 늘면 여기에 한 줄만 추가한다.
TRACKS = {
    "t1": t1_approval_line.run,
    "t1c": t1c_approval_predict.run,
    "t1d": t1d_draft.run,
    "t4": t4_gazette_ocr.run,
    "t5": t5_gazette_summary.run,
}


@app.callback()
def main() -> None:
    """트랙 단위로 벤치마크를 돌린다. 커맨드는 `run` 하나로 시작한다."""


@app.command()
def run(
    track: str = typer.Argument(..., help="실행할 트랙 이름 (예: t4)"),
    dpi: int = typer.Option(200, help="렌더링 해상도"),
    pages: int | None = typer.Option(None, "--pages", help="문서당 앞에서부터 볼 페이지 수"),
    docs: int | None = typer.Option(None, "--docs", help="훑을 문서 수"),
    only: str | None = typer.Option(None, "--only", help="파일명에 이 문자열이 든 PDF만 (예: 21317)"),
    select: str = typer.Option("first", "--select", help="페이지 고르기: first | mixed(유형 섞기)"),
    model: str | None = typer.Option(None, "--model", help="t1c/t1d 에서 쓸 ollama 생성 모델"),
    embed_model: str | None = typer.Option(None, "--embed-model", help="t1d 임베딩 모델"),
    drafts: int = typer.Option(3, "--drafts", help="t1d 케이스당 초안 개수"),
    resample: int = typer.Option(0, "--resample", help="t1c 재표집 검증 라운드 수"),
    qa: bool = typer.Option(False, "--qa", help="t5 자연어 질의응답 RAG 실행"),
) -> None:
    """트랙 하나를 collect → parse → evaluate 순서로 실행한다."""
    if track not in TRACKS:
        raise typer.BadParameter(f"모르는 트랙: {track} (가능: {', '.join(TRACKS)})")

    if track == "t5":
        t5_gazette_summary.report(
            TRACKS[track](
                model=model, embed_model=embed_model, sample=docs or 24, qa=qa
            )
        )
        return

    if track == "t1d":
        t1d_draft.report(
            TRACKS[track](model=model, embed_model=embed_model, n_drafts=drafts)
        )
        return

    if track == "t1c":
        t1c_approval_predict.report(
            TRACKS[track](model=model, max_docs=docs, resample=resample)
        )
        return

    if track == "t1":
        t1_approval_line.report(TRACKS[track](max_docs=docs))
        return

    report = TRACKS[track](
        dpi=dpi, max_pages=pages, max_docs=docs, only=only, select=select
    )
    summary = report["summary"]

    typer.echo(f"[{track}] {report['ocr']} vs {report['ground_truth']}")
    typer.echo(f"       정규화: {report['normalization']}")
    for doc in report["documents"]:
        sel = doc["page_selection"]
        typer.echo(
            f"  {doc['pdf']} — 전체 {doc['doc_pages']}p 중 {doc['scored_pages']}p 채점"
            f" [{sel['strategy']}] ({doc['parse_sec']}s)"
        )
        typer.echo(f"    고른 페이지: {doc['page_selection'].get('doc_type_counts', '')}")
        for page in doc["pages"]:
            if "cer" not in page:
                typer.echo(f"    p{page['page']}: 건너뜀 ({page['skipped']})")
                continue
            typer.echo(
                f"    p{page['page']:>3} [{page['page_type']:9}] CER {page['cer']:.4f}"
                f" | 리더제외 {page['cer_no_leader']:.4f}"
                f" | 공백제외 {page['cer_nospace']:.4f}"
                f" | 겹침 {page['char_overlap']:.4f}"
            )
        for t, m in doc["by_page_type"].items():
            typer.echo(
                f"    · {t:9} ({m['pages']:2}p): CER {m['cer']:.4f}"
                f" | 리더제외 {m['cer_no_leader']:.4f} | 겹침 {m['char_overlap']:.4f}"
            )
        overall = doc["overall"]
        typer.echo(
            f"    = 문서 전체: CER {overall['cer']:.4f}"
            f" | 리더제외 {overall['cer_no_leader']:.4f}"
            f" | 공백제외 {overall['cer_nospace']:.4f}"
            f" | 겹침 {overall['char_overlap']:.4f} | WER {overall['wer']:.4f}"
        )

    for skip in report["skipped"]:
        typer.echo(f"  건너뜀: {skip['pdf']} ({skip['reason']})")

    typer.echo(
        f"  == 문서 {summary['documents']}건 / {summary['pages']}페이지 "
        f"평균 CER {summary['cer_mean']:.4f} | 리더제외 {summary['cer_no_leader_mean']:.4f} "
        f"| 공백제외 {summary['cer_nospace_mean']:.4f} | 겹침 {summary['char_overlap_mean']:.4f} "
        f"({summary['elapsed_sec']}s)"
    )
    typer.echo(f"  → {report['results_path']}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="바인드 주소. 기본은 localhost 전용"),
    port: int = typer.Option(8000, help="포트"),
) -> None:
    """데모 API를 띄운다. Swagger 는 /swagger 에 있다."""
    import uvicorn

    typer.echo(f"Swagger: http://{host}:{port}/swagger")
    uvicorn.run("who_approved_this.api.app:app", host=host, port=port, log_level="info")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="바인드 주소. 기본은 localhost 전용"),
    port: int = typer.Option(7860, help="포트"),
) -> None:
    """Gradio 데모 화면을 띄운다."""
    from who_approved_this import ui as ui_module

    ui_module.main(host, port)


if __name__ == "__main__":
    app()
