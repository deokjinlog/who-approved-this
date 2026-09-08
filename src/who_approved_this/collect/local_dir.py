"""로컬 디렉터리 수집기.

수집 자동화 전 단계. 손으로 받아둔 원문 PDF와 정답 텍스트를
:data:`~who_approved_this.config.DATA_ROOT` 아래에서 그대로 읽는다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from who_approved_this.collect.base import Collector


class LocalPairCollector(Collector):
    """``<이름>.pdf`` 와 같은 이름의 ``<이름>.txt`` 를 쌍으로 내놓는 수집기.

    ``ground_truth`` 는 ``{"text": <정답 전문>, "doc_id": <파일 stem>}``.
    ``.txt`` 가 없는 PDF는 건너뛰고 :attr:`skipped` 에 남긴다.
    """

    def __init__(self, data_dir: Path, name: str = "local-dir") -> None:
        self.data_dir = data_dir
        self.name = name
        self.skipped: list[Path] = []

    def items(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        """디렉터리의 PDF를 이름순으로 훑어 ``(pdf_path, ground_truth)`` 를 yield 한다."""
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"데이터 디렉터리가 없다: {self.data_dir}")

        self.skipped = []
        for pdf_path in sorted(self.data_dir.glob("*.pdf")):
            truth_path = pdf_path.with_suffix(".txt")
            if not truth_path.is_file():
                self.skipped.append(pdf_path)
                continue
            yield pdf_path, {
                "doc_id": pdf_path.stem,
                "text": truth_path.read_text(encoding="utf-8"),
            }
