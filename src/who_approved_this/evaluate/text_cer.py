"""텍스트 대조 — jiwer로 CER/WER 산출.

대조 전 정규화를 함수 하나로 떼어 두고, **정규화 전/후를 둘 다** 잰다.
정규화가 점수를 얼마나 밀어 올리는지 보이지 않으면, 낮은 CER이 OCR이 잘한 건지
정규화가 덮은 건지 구분할 수 없다.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import jiwer

from who_approved_this.evaluate.base import Evaluator

_WS = re.compile(r"\s+")

#: 결과 JSON에 남길 정규화 방식 설명. 점수를 재현하려면 이게 같아야 한다.
NORMALIZATION = "unicode-nfc + whitespace-collapse + strip"


def normalize(text: str) -> str:
    """유니코드 NFC로 모으고, 모든 공백을 한 칸으로 접고, 앞뒤를 턴다.

    * NFC — 한글 자모 분리(NFD) 표기를 완성형으로 모은다. macOS 파일/텍스트에서
      섞여 들어오면 글자는 같은데 코드포인트가 달라 CER이 부풀려진다.
    * 공백 압축 — 줄바꿈·단 나눔 차이로 점수가 흔들리지 않게.
    """
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


class TextCEREvaluator(Evaluator):
    """OCR 전문과 정답 전문을 대조해 CER/WER를 낸다.

    입력
        * ``parsed: str`` — 파서가 돌려준 전문.
        * ``ground_truth: {"text": str}`` — 정답 전문.
    출력
        ``{"cer", "wer", "cer_raw", "wer_raw", "ref_chars", "hyp_chars"}``.
        ``*_raw`` 는 정규화하지 않은 원본끼리 잰 값, 나머지는 :func:`normalize`
        를 거친 값. WER은 공백으로 자르므로 한국어에서는 어절 단위다.
    """

    name = "text-cer"

    #: 결과표·JSON에 그대로 실리는 정규화 방식.
    normalization = NORMALIZATION

    def evaluate(
        self, parsed: str | dict[str, Any], ground_truth: dict[str, Any]
    ) -> dict[str, float]:
        """``parsed`` 를 ``ground_truth["text"]`` 와 대조한 지표 dict 를 돌려준다."""
        if not isinstance(parsed, str):
            raise TypeError(f"{type(self).__name__} 은 str 결과만 받는다: {type(parsed)}")
        if "text" not in ground_truth:
            raise KeyError("ground_truth 에 'text' 키가 필요하다")

        ref_raw = str(ground_truth["text"])
        ref, hyp = normalize(ref_raw), normalize(parsed)
        if not ref:
            raise ValueError("정답 텍스트가 비어 있다")

        return {
            "cer": float(jiwer.cer(ref, hyp)),
            "wer": float(jiwer.wer(ref, hyp)),
            "cer_raw": float(jiwer.cer(ref_raw, parsed)),
            "wer_raw": float(jiwer.wer(ref_raw, parsed)),
            "ref_chars": float(len(ref)),
            "hyp_chars": float(len(hyp)),
        }
