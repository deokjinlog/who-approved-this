"""정보공개포털(open.go.kr) 원문 결재문서 수집기 — 로컬 배치본.

``DATA_ROOT/t1/`` 에 손으로 받아둔 PDF와 ``metadata.tsv`` 를 짝지어 내놓는다.
수집 절차 자체는 같은 디렉터리의 ``download_recipe.md`` 에 기록돼 있다.

정답의 범위 — T1a / T1b 를 가르는 이유
--------------------------------------
포털 메타데이터에는 **결재선이 없다.** ``결재정보`` 컬럼은 10건 전부
``"(화면에 표시 항목 없음)"`` 이고, 사람 이름은 ``담당자``(기안자)까지만 나온다.

* **T1a** — 제목·생산기관·담당부서·담당자·생산일자·문서번호 6개 필드. TSV가 정답이다.
* **T1b** — 결재선(기안/검토/결재/협조의 직위·이름). **정답이 없다.**
  텍스트 레이어에서 뽑은 값을 잠정 기준으로 두고, 나중에 OCR·VLM 추출값과 대조한다.

개인정보 취급
-------------
``담당자`` 는 실명이다. 이 수집기는 정답 dict 에 값을 담아 평가 단계로 넘기지만,
**결과 JSON과 콘솔에는 값을 남기지 않는다**(일치 여부 bool 과 직위만 기록).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from who_approved_this.collect.base import Collector

#: T1a 채점 대상 6개 필드. 키는 metadata.tsv 헤더 그대로 쓴다.
T1A_FIELDS = ("제목", "생산기관", "담당부서", "담당자", "생산일자", "문서번호")


class OpenGoKrLocalCollector(Collector):
    """``metadata.tsv`` 의 각 행과 같은 이름의 PDF를 쌍으로 내놓는다.

    ``ground_truth`` 는 TSV 한 행을 그대로 담은 dict 에
    ``doc_id``(저장파일명 stem)를 더한 것이다. 컬럼명은 헤더 그대로 둔다.
    """

    name = "opengokr-local"

    def __init__(self, data_dir: Path, max_docs: int | None = None) -> None:
        self.data_dir = data_dir
        self.max_docs = max_docs
        self.skipped: list[tuple[str, str]] = []

    def items(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        """``(pdf_path, ground_truth)`` 를 TSV 순서대로 yield 한다."""
        tsv = self.data_dir / "metadata.tsv"
        if not tsv.is_file():
            raise FileNotFoundError(f"metadata.tsv 가 없다: {tsv}")

        self.skipped = []
        yielded = 0
        with tsv.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if self.max_docs is not None and yielded >= self.max_docs:
                    break
                pdf_path = self.data_dir / row["저장파일명"]
                if not pdf_path.is_file():
                    # 실명이 들어 있는 제목 대신 저장파일명으로만 기록한다.
                    self.skipped.append((row["저장파일명"], "PDF 없음"))
                    continue
                yield pdf_path, {"doc_id": pdf_path.stem, **row}
                yielded += 1
