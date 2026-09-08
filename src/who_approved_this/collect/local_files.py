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
from who_approved_this.collect.page_type import classify_page, page_stats


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
        select: str = "first",
    ) -> None:
        self.data_dir = data_dir
        self.name = name
        #: 파일명에 이 문자열이 들어간 PDF만 본다. ``None`` 이면 전부.
        #: 큰 문서를 실수로 돌리지 않으려고 대상을 좁힐 때 쓴다.
        self.only = only
        #: 페이지 고르는 방식. ``"first"`` 는 앞에서부터, ``"mixed"`` 는
        #: 유형(toc/table/body)이 섞이도록 고른다. 앞 20장만 보면 목차·공포문에
        #: 치우쳐 문서 전체를 대표하지 못한다.
        self.select = select
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
            truth = self._read(pdf_path)
            if not any(t.strip() for t in truth["pages"]):
                # 텍스트 레이어가 없으면 이 수집기로는 정답을 만들 수 없다.
                self.skipped.append((pdf_path, "텍스트 레이어 없음(스캔 전용으로 보임)"))
                continue
            yield pdf_path, truth
            yielded += 1

    def _read(self, pdf_path: Path) -> dict[str, Any]:
        """고른 페이지의 텍스트 레이어와 유형 꼬리표를 함께 읽는다.

        ``sort=True`` 로 **시각적 읽는 순서**(위→아래, 왼→오른쪽)를 쓴다.
        pymupdf 의 기본값은 PDF 안에 그려진 블록 순서라 사람이 읽는 순서와 다르고,
        OCR은 화면에 보이는 순서로 읽으므로 그대로 두면 글자를 다 맞혀도 순서 차이가
        CER로 잡힌다. 실제로 제21317호 20페이지에서 이 한 줄이 CER 0.2755 → 0.1105 을 갈랐다.
        """
        with pymupdf.open(pdf_path) as doc:
            numbers, selection = self._choose_pages(doc)
            stats = [page_stats(doc[n - 1]) for n in numbers]
            texts = [doc[n - 1].get_text(sort=True) for n in numbers]
            return {
                "doc_id": pdf_path.stem,
                "text": "\n\n".join(texts),
                "pages": texts,
                "page_numbers": numbers,
                "page_stats": stats,
                "page_count": len(numbers),
                "doc_pages": doc.page_count,
                "page_selection": selection,
            }

    def _choose_pages(self, doc: pymupdf.Document) -> tuple[list[int], dict[str, Any]]:
        """볼 페이지 번호(1부터)와, 어떻게 골랐는지 설명을 함께 돌려준다."""
        n_want = doc.page_count if self.max_pages is None else min(self.max_pages, doc.page_count)

        if self.select != "mixed":
            return list(range(1, n_want + 1)), {
                "strategy": "first",
                "description": f"앞에서부터 {n_want}페이지",
                "doc_pages": doc.page_count,
            }

        # 전체 페이지를 유형별로 나눈 뒤 유형을 돌아가며 한 장씩 집는다.
        # 문서에 실제로 존재하는 유형 비율과 무관하게 각 유형이 표본에 들어온다.
        buckets: dict[str, list[int]] = {}
        for i in range(doc.page_count):
            buckets.setdefault(classify_page(doc[i]), []).append(i + 1)

        totals = {t: len(v) for t, v in sorted(buckets.items())}
        picked: list[int] = []
        order = sorted(buckets)
        while len(picked) < n_want and any(buckets[t] for t in order):
            for t in order:
                if not buckets[t] or len(picked) >= n_want:
                    continue
                picked.append(buckets[t].pop(0))
        picked.sort()
        return picked, {
            "strategy": "mixed-by-type",
            "description": (
                "전체 페이지를 유형(toc/table/body_1col/body_2col)으로 분류한 뒤 "
                f"유형을 돌아가며 한 장씩 집어 {len(picked)}페이지를 골랐다. "
                "앞 N장만 보면 목차·공포문에 치우쳐 문서를 대표하지 못한다."
            ),
            "doc_pages": doc.page_count,
            "doc_type_counts": totals,
        }
