"""텍스트 대조 — jiwer로 CER/WER 산출.

대조 전 정규화를 단계별 함수로 떼어 두고, **정규화 전/후를 둘 다** 잰다.
정규화가 점수를 얼마나 밀어 올리는지 보이지 않으면, 낮은 CER이 OCR이 잘한 건지
정규화가 덮은 건지 구분할 수 없다.

왜 정규화가 필요한가 (제21317호 20페이지 실측에서 확인된 상위 오류)
--------------------------------------------------------------------
관보 텍스트 레이어와 OCR 출력의 차이 대부분은 **글자 오인식이 아니라 표기 관습**이었다.

* 공백 — 정답 글자의 25~30%가 공백. 관보는 자간을 공백으로 벌린다
  ("대 통 령", "관    보"). OCR은 그 간격을 그대로 재현하지 않는다.
* 따옴표 — 정답은 둥근 따옴표(``“ ”``), OCR은 곧은 따옴표(``"``).
* 가운뎃점 — 정답은 ``ᆞ``(U+119E, 아래아)를 구분점으로 쓰는데 OCR은 ``•``(U+2022)로 읽는다.
* 점선 리더 — 목차 줄을 채우는 ``·······`` 는 내용이 아니라 조판이다.

이걸 정규화하지 않으면 CER이 OCR 품질이 아니라 조판 관습을 재게 된다.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

import jiwer

from who_approved_this.evaluate.base import Evaluator

_WS = re.compile(r"\s+")

#: 둥근/곧은 따옴표를 한 글자로 모은다.
_QUOTES = {
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
}

#: 괄호 변종을 대표 글자로 모은다.
#: 관보는 법령명에 ``「 」``, 편 제목에 ``【 】`` 를 쓰는데 OCR 은 이를
#: ``[ ]`` · ``『 』`` 등으로 읽는 일이 잦다. 같은 기호를 다르게 적은 것뿐이라
#: 표기 관습으로 보고 통일한다(실측 12페이지에서 「」→『』 5건, 【】→[] 7건).
_BRACKETS = {
    "【": "[", "】": "]", "〔": "[", "〕": "]", "［": "[", "］": "]",
    "「": "[", "」": "]", "『": "[", "』": "]",
    "〈": "<", "〉": ">", "《": "<", "》": ">",
}

#: 가운뎃점 변종을 ``·``(U+00B7) 하나로 모은다.
#: ``ᆞ``(U+119E)는 관보 텍스트 레이어가 실제로 쓰는 아래아이고,
#: ``•``(U+2022)는 OCR이 같은 자리에서 내놓는 글자다.
_INTERPUNCTS = "·ᆞ•∙⋅・·‧"

#: 점 3개 이상이 이어지면 내용이 아니라 목차 점선 리더로 본다.
_LEADERS = re.compile(f"[{_INTERPUNCTS}\\s]*[{_INTERPUNCTS}]{{3,}}[{_INTERPUNCTS}\\s]*")

#: 기본 정규화. 리더 제거는 **포함하지 않는다** — 지표를 유리하게 만드는 조작이
#: 기본값에 숨지 않도록, 리더를 뺀 값은 ``cer_no_leader`` 로 따로 기록한다.
NORMALIZATION = "nfc + quotes + brackets + interpunct + whitespace-collapse"

#: 리더 제거까지 적용한 정규화(``cer_no_leader`` 용).
NORMALIZATION_NO_LEADER = NORMALIZATION + " + leaders-removed"


def unify_punct(text: str) -> str:
    """따옴표·괄호·가운뎃점 변종을 대표 글자 하나로 모은다.

    **모양이 다른 같은 기호**만 모은다. ``○`` → ``·`` 처럼 OCR 이 실제로 잘못 읽은
    것은 손대지 않는다 — 그건 재려는 대상이지 표기 관습이 아니다.
    """
    out = text.translate(str.maketrans(_QUOTES))
    out = out.translate(str.maketrans(_BRACKETS))
    return out.translate(str.maketrans(_INTERPUNCTS, "·" * len(_INTERPUNCTS)))


def drop_leaders(text: str) -> str:
    """목차 점선 리더(점 3개 이상 연속)를 공백 하나로 바꾼다.

    구분점으로 쓰이는 단독 ``·`` 는 건드리지 않는다(연속 3개 이상만 리더로 본다).
    """
    return _LEADERS.sub(" ", text)


def normalize(text: str, *, no_leader: bool = False) -> str:
    """NFC → 부호 통일 → (선택) 리더 제거 → 공백 압축 순으로 정리한다.

    NFC는 한글 자모 분리(NFD) 표기를 완성형으로 모은다. macOS에서 섞여 들어오면
    글자는 같은데 코드포인트가 달라 CER이 부풀려진다.

    ``no_leader=True`` 면 목차 점선 리더를 지운다. 기본값은 ``False`` — 리더도
    정답에 있는 글자이므로, 지운 값은 별도 지표로만 보고한다.
    """
    out = unicodedata.normalize("NFC", text)
    out = unify_punct(out)
    if no_leader:
        out = drop_leaders(out)
    return _WS.sub(" ", out).strip()


def char_overlap(ref: str, hyp: str) -> float:
    """글자 다중집합 겹침 — 순서를 무시하고 "글자를 맞혔는지"만 본다.

    ``|ref ∩ hyp| / |ref|`` (공백 제외). 1에 가까운데 CER이 높으면 글자는 맞고
    **순서만 다르다**는 뜻이다. 읽는 순서가 성능과 무관한 과제(T1의 필드 추출처럼
    어디서 뽑았든 값만 맞으면 되는 경우)에 CER보다 가까운 지표다.
    """
    ref_c, hyp_c = Counter(strip_spaces(ref)), Counter(strip_spaces(hyp))
    total = sum(ref_c.values())
    if not total:
        return 0.0
    return sum((ref_c & hyp_c).values()) / total


def strip_spaces(text: str) -> str:
    """공백을 전부 뺀다.

    한국어는 띄어쓰기가 흔들려도 사람이 읽는 내용은 같으므로("결재 라인" vs
    "결재라인"), 순수 글자 인식률을 보려면 공백을 빼고 재는 쪽이 표준에 가깝다.
    """
    return _WS.sub("", text)


class TextCEREvaluator(Evaluator):
    """OCR 전문과 정답 전문을 대조해 CER/WER를 낸다.

    입력
        * ``parsed: str`` — 파서가 돌려준 전문.
        * ``ground_truth: {"text": str}`` — 정답 전문.
    출력
        ``{"cer", "wer", "cer_no_leader", "cer_nospace", "char_overlap",
        "cer_raw", "wer_raw", "ref_chars", "hyp_chars"}``.

        * ``cer`` / ``wer`` — :func:`normalize` 기본값을 거친 값(리더 포함).
        * ``cer_no_leader`` — 목차 점선 리더를 지우고 잰 값.
        * ``cer_nospace`` — 공백까지 뺀 값. **한국어 OCR 정확도의 주 지표.**
        * ``char_overlap`` — 순서를 무시한 글자 겹침(:func:`char_overlap`).
        * ``*_raw`` — 정규화하지 않은 원본끼리 잰 값. 정규화가 얼마나 덮었는지 보려고 함께 남긴다.

        WER은 공백으로 자르므로 한국어에서는 어절 단위이고, 띄어쓰기 변동에
        과민하다. 참고 지표로만 본다.
    """

    name = "text-cer"

    #: 결과표·JSON에 그대로 실리는 정규화 방식.
    normalization = NORMALIZATION
    #: ``cer_no_leader`` 에 적용된 정규화 방식.
    normalization_no_leader = NORMALIZATION_NO_LEADER

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

        ref_nl = normalize(ref_raw, no_leader=True)
        hyp_nl = normalize(parsed, no_leader=True)
        ref_ns, hyp_ns = strip_spaces(ref), strip_spaces(hyp)
        return {
            "cer": float(jiwer.cer(ref, hyp)),
            "wer": float(jiwer.wer(ref, hyp)),
            "cer_no_leader": float(jiwer.cer(ref_nl, hyp_nl)) if ref_nl else 0.0,
            "cer_nospace": float(jiwer.cer(ref_ns, hyp_ns)) if ref_ns else 0.0,
            "char_overlap": float(char_overlap(ref, hyp)),
            "cer_raw": float(jiwer.cer(ref_raw, parsed)),
            "wer_raw": float(jiwer.wer(ref_raw, parsed)),
            "ref_chars": float(len(ref)),
            "hyp_chars": float(len(hyp)),
        }
