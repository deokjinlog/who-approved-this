"""T1b 결재선 대조 — ``approval_gold.tsv`` 와 맞춰본다.

정답의 출처
------------
포털 메타데이터에는 결재선이 없다. 그래서 gold 는 PDF 결재란을 **사람이 직접 보고**
만든 것이고, 리포 밖 ``DATA_ROOT/t1/approval_gold.tsv`` 에 둔다(실명 포함).

gold 의 한계도 데이터에 적어 두었다.

* 문서에 ``기안/검토/결재`` 라는 **역할 이름이 인쇄돼 있지 않다.** 직위와 이름만 있다.
  역할은 칸 위치로 추정했고 ``note`` 에 "역할 추정"으로 표시했다.
* 사람이 읽을 수 없는 칸은 ``name`` 을 ``?`` 로 두었다. 이 칸은 **이름 채점에서 빼고**
  따로 센다. "사람도 못 읽는 건 모델도 못 읽는다"는 기준선이다.

개인정보
--------
비교는 이 모듈 안에서만 하고, 돌려주는 값에 이름·직위 문자열은 담지 않는다.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any

from who_approved_this.evaluate.base import Evaluator

ROLES = ("기안", "검토", "결재", "협조")

#: 사람이 못 읽은 칸 표시.
UNREADABLE = "?"

#: 칸이 아예 없다는 표시(협조란 없음 등).
EMPTY = "-"

_STRIP = re.compile(r"[\s()\[\]·.\-]")


def normalize_token(value: str) -> str:
    """직위·이름 비교용 정규화. 공백·괄호·가운뎃점을 걷는다."""
    return _STRIP.sub("", unicodedata.normalize("NFC", str(value))).strip()


def load_orgs(path: Path) -> dict[str, str]:
    """``approval_gold.tsv`` 의 ``org`` 열을 ``{저장파일명: 조직}`` 으로 읽는다.

    포털 메타데이터의 ``담당부서`` 는 기관 간에 재사용되므로("총무과"는 거의 모든
    기관에 있다) 조직을 식별하지 못한다. 그래서 문서 하단 주소·발신기관을 보고
    사람이 붙인 조직 키를 gold 에 함께 둔다.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["저장파일명"]: r.get("org", "") for r in csv.DictReader(fh, delimiter="\t")}


def load_gold(path: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    """``approval_gold.tsv`` 를 ``{저장파일명: {역할: [칸, ...]}}`` 로 읽는다."""
    gold: dict[str, dict[str, list[dict[str, str]]]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            doc = gold.setdefault(row["저장파일명"], {r: [] for r in ROLES})
            if row["title"] == EMPTY and row["name"] == EMPTY:
                continue  # "협조란 없음" 표시 행 — 기대 칸 수 0
            doc[row["role"]].append(
                {"order": row["order"], "title": row["title"], "name": row["name"],
                 "note": row.get("note", "")}
            )
    return gold


class ApprovalMatchEvaluator(Evaluator):
    """추출 결재선을 gold 와 대조해 역할별 일치 지표를 낸다.

    출력 지표(전부 수치, 값 없음)::

        role_<역할>_gold_cells      gold 칸 수
        role_<역할>_pred_cells      추출 칸 수
        role_<역할>_count_match     칸 수 일치 여부(0/1)
        role_<역할>_title_hits      직위 일치 칸 수
        role_<역할>_name_hits       이름 일치 칸 수(미판독 칸 제외)
        role_<역할>_name_scored     이름 채점 대상 칸 수
        unreadable_cells            gold 가 "?" 로 둔 칸 수
        title_accuracy / name_accuracy / cell_count_match
    """

    name = "approval-match"

    def __init__(self, gold: dict[str, dict[str, list[dict[str, str]]]]) -> None:
        self.gold = gold
        #: 마지막 evaluate 의 실패 목록. ``[(역할, 사유)]`` — 값은 담지 않는다.
        self.failures: list[tuple[str, str]] = []

    def evaluate(
        self, parsed: str | dict[str, Any], ground_truth: dict[str, Any]
    ) -> dict[str, float]:
        """추출 결과를 gold 와 대조한다. ``ground_truth`` 는 ``{"저장파일명": ...}``."""
        if not isinstance(parsed, dict):
            raise TypeError(f"{type(self).__name__} 은 dict 결과만 받는다: {type(parsed)}")
        key = ground_truth["저장파일명"]
        if key not in self.gold:
            raise KeyError(f"gold 에 없는 문서: {key}")

        self.failures = []
        want = self.gold[key]
        got: dict[str, list[dict[str, str]]] = {r: [] for r in ROLES}
        for cell in parsed.get("cells", []):
            got.setdefault(cell["role"], []).append(cell)

        scores: dict[str, float] = {}
        t_hit = t_tot = n_hit = n_tot = unread = 0
        count_ok = 0
        for role in ROLES:
            g, p = want[role], got.get(role, [])
            scores[f"role_{role}_gold_cells"] = float(len(g))
            scores[f"role_{role}_pred_cells"] = float(len(p))
            same = len(g) == len(p)
            scores[f"role_{role}_count_match"] = float(same)
            count_ok += same
            if not same:
                self.failures.append((role, f"칸 수 불일치 (gold {len(g)} / 추출 {len(p)})"))

            th = nh = ns = 0
            for i, gc in enumerate(g):
                pc = p[i] if i < len(p) else None
                t_tot += 1
                if pc and normalize_token(pc["title"]) == normalize_token(gc["title"]):
                    th += 1
                else:
                    self.failures.append((role, f"{i + 1}번 칸 직위 틀림" if pc else f"{i + 1}번 칸 누락"))
                if gc["name"] == UNREADABLE:
                    unread += 1
                    continue
                ns += 1
                if pc and normalize_token(pc["name"]) == normalize_token(gc["name"]):
                    nh += 1
                else:
                    self.failures.append((role, f"{i + 1}번 칸 이름 틀림"))
            scores[f"role_{role}_title_hits"] = float(th)
            scores[f"role_{role}_name_hits"] = float(nh)
            scores[f"role_{role}_name_scored"] = float(ns)
            t_hit += th
            n_hit += nh
            n_tot += ns

        scores["unreadable_cells"] = float(unread)
        scores["title_accuracy"] = t_hit / t_tot if t_tot else 0.0
        scores["name_accuracy"] = n_hit / n_tot if n_tot else 0.0
        scores["cell_count_match"] = count_ok / len(ROLES)
        return scores
