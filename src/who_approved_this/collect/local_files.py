"""로컬 PDF 수집기 — 정답을 같은 PDF의 **텍스트 레이어**에서 뽑는다.

이 수집기가 만드는 벤치마크는 "**텍스트 레이어 vs OCR**" 비교다.
정답(ground truth)은 외부 공개 텍스트가 아니라 PDF 안에 이미 박혀 있는
텍스트 레이어(``pymupdf`` 의 ``page.get_text()``)이고, 가설은 하나다.

    같은 PDF를 200dpi로 렌더링해 OCR로 다시 읽으면
    원래 텍스트 레이어를 얼마나 되살리는가.

그래서 여기서 나오는 CER은 **OCR 엔진의 절대 정확도가 아니다**. 한계가 분명하다.

* 텍스트 레이어 자체가 정답이라는 보장이 없다. 스캔본에 얹힌 부정확한 OCR
  레이어라면 정답이 틀린 채로 점수가 나온다.
* 텍스트 레이어가 없는 순수 스캔 PDF는 아예 잴 수 없다(건너뛴다).
* 읽는 순서(단 나눔·표)가 텍스트 레이어와 OCR에서 다르면 글자를 다 맞혀도
  CER이 올라간다.

진짜 관보 OCR 정확도는 국가법령정보 등 **공개 구조화 텍스트**를 정답으로 놓고
재야 한다(:class:`~who_approved_this.collect.local_dir.LocalPairCollector`).
이 수집기는 그 전에 파이프라인을 세로로 관통시키고 배치 난이도를 낮추는 쪽이다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymupdf

from who_approved_this.collect.base import Collector


class TextLayerCollector(Collector):
    """``DATA_ROOT/<track>/*.pdf`` 를 훑어 텍스트 레이어를 정답으로 내놓는다.

    ``ground_truth`` 는
    ``{"doc_id": <파일 stem>, "text": <전문>, "pages": [<페이지별 텍스트>], "page_count": int}``.

    텍스트 레이어가 비어 있는(=스캔 전용) PDF는 정답이 없으므로 건너뛰고
    :attr:`skipped` 에 ``(경로, 사유)`` 로 남긴다.
    """

    def __init__(
        self,
        data_dir: Path,
        name: str = "local-files-textlayer",
        max_pages: int | None = None,
        max_docs: int | None = None,
        only: str | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.name = name
        #: 파일명에 이 문자열이 들어간 PDF만 본다. ``None`` 이면 전부.
        #: 큰 문서를 실수로 돌리지 않으려고 대상을 좁힐 때 쓴다.
        self.only = only
        #: 문서당 앞에서부터 볼 페이지 수. ``None`` 이면 전체.
        self.max_pages = max_pages
        #: 훑을 문서 수. ``None`` 이면 전체.
        self.max_docs = max_docs
        self.skipped: list[tuple[Path, str]] = []

    def items(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        """PDF를 이름순으로 훑어 ``(pdf_path, ground_truth)`` 를 yield 한다."""
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"데이터 디렉터리가 없다: {self.data_dir}")

        self.skipped = []
        yielded = 0
        for pdf_path in sorted(self.data_dir.glob("*.pdf")):
            if self.only is not None and self.only not in pdf_path.name:
                continue
            if self.max_docs is not None and yielded >= self.max_docs:
                break
            pages = self._page_texts(pdf_path)
            if not any(p.strip() for p in pages):
                # 텍스트 레이어가 없으면 이 수집기로는 정답을 만들 수 없다.
                self.skipped.append((pdf_path, "텍스트 레이어 없음(스캔 전용으로 보임)"))
                continue
            yield pdf_path, {
                "doc_id": pdf_path.stem,
                "text": "\n\n".join(pages),
                "pages": pages,
                "page_count": len(pages),
            }
            yielded += 1

    def _page_texts(self, pdf_path: Path) -> list[str]:
        """앞에서부터 :attr:`max_pages` 장의 텍스트 레이어를 페이지별로 읽는다."""
        with pymupdf.open(pdf_path) as doc:
            limit = doc.page_count if self.max_pages is None else min(self.max_pages, doc.page_count)
            return [doc[i].get_text() for i in range(limit)]
