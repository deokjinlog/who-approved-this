"""T5 요약 대조 — 정답 요약이 없을 때 무엇을 잴 수 있나.

관보에는 사람이 쓴 요약문이 없다. 대신 **공식 제명**이 있다.
제명은 "이 문서가 무엇인가"를 기관이 직접 붙인 한 줄이라, 요약이 그 핵심을
담았는지를 재는 기준으로 쓸 수 있다.

여기에 T1d 에서 배운 것을 더한다 — **요약이 지어내지 않았는지**가 정확도만큼 중요하다.
초안 실험에서 모델은 형식을 정확히 흉내 내면서 문서번호를 6/7 지어냈다.
요약에서도 같은 일이 일어나는지 본다.

지표
    ``title_recall``    제명의 내용어가 요약에 몇 개 들어갔나
    ``grounded_ratio``  요약에 등장한 **숫자 토큰**(조문·호수·날짜) 중 본문에 실재하는 비율
    ``compression``     요약 길이 / 본문 길이
"""

from __future__ import annotations

import re
from typing import Any

from who_approved_this.evaluate.base import Evaluator

#: 제명에서 내용어로 보지 않을 낱말. 거의 모든 제명에 들어가 변별력이 없다.
TITLE_STOP = {
    "관한", "등의", "등에", "대한", "위한", "및", "일부개정", "일부개정령",
    "일부개정법률", "시행령", "시행규칙", "법률", "규정", "고시", "공고", "개정",
}

#: 내용어 후보 — 한글 2자 이상.
WORD = re.compile(r"[가-힣]{2,}")

#: 숫자 토큰 — 조문·호수·연도·금액 등. 요약이 지어내기 쉬운 자리다.
NUMBER = re.compile(r"\d[\d,\-.]*")


def title_words(title: str) -> list[str]:
    """제명에서 변별력 있는 내용어만 남긴다."""
    return [w for w in WORD.findall(title) if w not in TITLE_STOP and len(w) >= 2]


def grounded_numbers(summary: str, body: str) -> tuple[int, int]:
    """요약의 숫자 토큰 중 본문에 실재하는 개수와 전체 개수."""
    body_flat = re.sub(r"[\s,]", "", body)
    found = [n for n in NUMBER.findall(summary) if len(re.sub(r"\D", "", n)) >= 2]
    ok = sum(1 for n in found if re.sub(r"[\s,]", "", n) in body_flat)
    return ok, len(found)


class SummaryMatchEvaluator(Evaluator):
    """요약을 **공식 제명**과 **본문 근거**로 대조한다.

    입력
        * ``parsed: str`` — 생성된 요약.
        * ``ground_truth: {"title": str, "body": str}`` — 공식 제명과 원문.
    출력
        ``{"title_recall", "title_words", "title_hits", "grounded_ratio",
        "numbers_total", "numbers_grounded", "compression", "summary_chars"}``.

        값은 담지 않는다. 제명·본문은 대조에만 쓰고 밖으로 내보내지 않는다.
    """

    name = "summary-match"

    def evaluate(
        self, parsed: str | dict[str, Any], ground_truth: dict[str, Any]
    ) -> dict[str, float]:
        """요약을 제명·본문과 대조한 지표 dict 를 돌려준다."""
        if not isinstance(parsed, str):
            raise TypeError(f"{type(self).__name__} 은 str 결과만 받는다: {type(parsed)}")
        summary = parsed.strip()
        title = str(ground_truth.get("title", ""))
        body = str(ground_truth.get("body", ""))

        words = title_words(title)
        hits = sum(1 for w in words if w in summary)
        ok, total = grounded_numbers(summary, body)
        return {
            "title_words": float(len(words)),
            "title_hits": float(hits),
            "title_recall": (hits / len(words)) if words else 0.0,
            "numbers_total": float(total),
            "numbers_grounded": float(ok),
            "grounded_ratio": (ok / total) if total else 1.0,
            "compression": (len(summary) / len(body)) if body else 0.0,
            "summary_chars": float(len(summary)),
        }
