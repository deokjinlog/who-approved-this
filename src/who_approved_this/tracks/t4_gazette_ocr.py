"""T4 — 관보 PDF 한국어 OCR 정량 벤치마크 (워밍업 트랙).

목적
    PDF를 200dpi로 렌더링해 로컬 OCR로 다시 읽었을 때, 원래 **텍스트 레이어**를
    얼마나 되살리는지 CER 숫자 하나로 확인한다. collect → parse → evaluate
    골격을 세로로 관통시키는 첫 트랙이다.

    정답이 텍스트 레이어이므로 이 점수는 **OCR 엔진의 절대 정확도가 아니다**.
    한계는 :mod:`who_approved_this.collect.local_files` 참고.

회사 프로젝트 환류
    이관 PDF 처리에서 로컬 OCR이 상용 DI를 어디까지 대체 가능한지.

파이프라인
    DATA_ROOT/t4/*.pdf
      → 텍스트 레이어 정답(pymupdf)
      → 200dpi PNG(DATA_ROOT/t4/rendered/<pdf명>/p{n}.png, 있으면 재사용)
      → Apple Vision OCR(ocrmac)
      → jiwer CER/WER (정규화 전/후)
      → results/t4/<YYYYMMDD-HHMM>.json

결과 JSON에는 pdf명·페이지 수·OCR 이름·정규화 방식·점수·소요시간만 남긴다.
본문 텍스트는 저장하지 않는다(CLAUDE.md: 실명·원문 금지).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from who_approved_this.collect.local_files import TextLayerCollector
from who_approved_this.config import track_data_dir, track_results_dir
from who_approved_this.evaluate.text_cer import TextCEREvaluator
from who_approved_this.parse.ocr_apple import AppleVisionOCRParser
from who_approved_this.parse.render import DEFAULT_DPI

TRACK = "t4"


def run(
    dpi: int = DEFAULT_DPI,
    max_pages: int | None = None,
    max_docs: int | None = None,
    only: str | None = None,
) -> dict[str, Any]:
    """T4를 collect → parse → evaluate 순서로 돌리고 결과 JSON을 남긴다.

    ``max_pages`` / ``max_docs`` 로 앞부분만 잘라 돌릴 수 있다(첫 실행용).
    """
    data_dir = track_data_dir(TRACK)
    results_dir = track_results_dir(TRACK)

    collector = TextLayerCollector(
        data_dir, max_pages=max_pages, max_docs=max_docs, only=only
    )
    parser = AppleVisionOCRParser(render_root=data_dir / "rendered", dpi=dpi)
    evaluator = TextCEREvaluator()

    started = time.perf_counter()
    documents = [
        _run_document(pdf_path, truth, parser, evaluator, max_pages)
        for pdf_path, truth in collector.items()
    ]
    elapsed = time.perf_counter() - started

    if not documents:
        raise FileNotFoundError(
            f"{data_dir} 에 텍스트 레이어를 가진 PDF가 없다(only={only!r}). "
            f"건너뜀: {[(p.name, why) for p, why in collector.skipped]}"
        )

    now = datetime.now()
    report = {
        "track": TRACK,
        "run_at": now.isoformat(timespec="seconds"),
        "ocr": parser.name,
        "ground_truth": "pdf-text-layer(pymupdf)",
        "evaluator": evaluator.name,
        "normalization": evaluator.normalization,
        "dpi": dpi,
        "max_pages": max_pages,
        "only": only,
        "documents": documents,
        "summary": {
            "documents": len(documents),
            "pages": sum(d["page_count"] for d in documents),
            "cer_mean": _mean(d["overall"]["cer"] for d in documents),
            "cer_nospace_mean": _mean(d["overall"]["cer_nospace"] for d in documents),
            "wer_mean": _mean(d["overall"]["wer"] for d in documents),
            "cer_raw_mean": _mean(d["overall"]["cer_raw"] for d in documents),
            "wer_raw_mean": _mean(d["overall"]["wer_raw"] for d in documents),
            "elapsed_sec": round(elapsed, 2),
        },
        "skipped": [{"pdf": p.name, "reason": why} for p, why in collector.skipped],
    }

    out_path = _results_path(results_dir, now)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["results_path"] = str(out_path)
    return report


def _results_path(results_dir: Path, now: datetime) -> Path:
    """``<YYYYMMDD-HHMM>.json``. 같은 분에 또 돌리면 초를 붙여 앞 결과를 덮지 않는다."""
    path = results_dir / f"{now:%Y%m%d-%H%M}.json"
    if path.exists():
        path = results_dir / f"{now:%Y%m%d-%H%M%S}.json"
    return path


def _run_document(
    pdf_path: Path,
    truth: dict[str, Any],
    parser: AppleVisionOCRParser,
    evaluator: TextCEREvaluator,
    max_pages: int | None,
) -> dict[str, Any]:
    """문서 하나를 OCR·채점해 페이지별 점수와 문서 전체 점수를 돌려준다."""
    t0 = time.perf_counter()
    ocr_pages = parser.parse_pages(pdf_path, max_pages=max_pages)
    parse_sec = time.perf_counter() - t0

    ref_pages: list[str] = truth["pages"]
    pages: list[dict[str, Any]] = []
    for n, (ref, hyp) in enumerate(zip(ref_pages, ocr_pages), start=1):
        if not ref.strip():
            # 텍스트 레이어가 빈 페이지는 정답이 없으니 채점에서 뺀다.
            pages.append({"page": n, "skipped": "텍스트 레이어 없음"})
            continue
        pages.append({"page": n, **_round(evaluator.evaluate(hyp, {"text": ref}))})

    scored = [p for p in pages if "cer" in p]
    if not scored:
        raise ValueError(f"{pdf_path.name}: 채점 가능한 페이지가 없다")

    # 전체 점수는 페이지 평균이 아니라 문서 전문끼리 한 번 더 대조해서 낸다.
    overall = evaluator.evaluate(
        "\n\n".join(ocr_pages), {"text": "\n\n".join(ref_pages)}
    )
    return {
        "pdf": pdf_path.name,
        "page_count": len(ocr_pages),
        "scored_pages": len(scored),
        "pages": pages,
        "overall": _round(overall),
        "parse_sec": round(parse_sec, 2),
    }


def _round(scores: dict[str, float]) -> dict[str, float]:
    """비율 지표는 소수 4자리로, 글자 수는 정수로 줄여 JSON을 읽기 쉽게 만든다."""
    return {
        k: int(v) if k.endswith("_chars") else round(v, 4) for k, v in scores.items()
    }


def _mean(values: Any) -> float:
    vals = list(values)
    return round(sum(vals) / len(vals), 4)
