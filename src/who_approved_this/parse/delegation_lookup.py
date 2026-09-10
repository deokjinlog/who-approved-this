"""전결 조회 — 문서가 별표1의 어느 사무인지 정하고 최소 전결 레벨을 돌려준다.

두 경로가 있다.

    집행 품의   제목이 품의·계약의뢰·구매·지급이면 (공통사항) 금액 구간표로 간다.
                구간표 선택(공사/용역/보조금/그 외)은 키워드, 금액은 라벨 붙은 값.
                **결정론이다.** 금액이 안 잡히면 모른다고 답한다.
    그 밖       부서 시트 + (공통사항) 사무명에서 BM25 로 후보 k개를 뽑고,
                모델이 있으면 그중 하나를 고르게 한다. 없으면 1위.

별표가 정하는 건 **최소** 전결권자다. 그보다 높은 사람이 결재해도 규정 위반이 아니다
(상향 결재). 그래서 gold 와 비교할 땐 "같다 / gold 가 더 높다 / gold 가 더 낮다(위반)"
세 갈래로 본다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from collections.abc import Callable

from rank_bm25 import BM25Okapi

from who_approved_this.tracks.t1d_retrieval import RRF_K, tokenize

_UNIT = {"천원": 1_000, "백만원": 1_000_000, "원": 1}


def load_table(path: Path) -> list[dict[str, Any]]:
    """행마다 ``rid``(표 안 순번)를 붙인다. gold 는 rid 로 행을 가리킨다."""
    rows = json.loads(path.expanduser().read_text(encoding="utf-8"))
    for i, r in enumerate(rows):
        r["rid"] = i
    return rows


def extract_labeled_amount(text: str, labels: list[str]) -> tuple[int | None, str | None]:
    """라벨 우선순위대로 금액을 찾는다. 공문은 라벨을 자간 벌려 쓴다("도 급 액")."""
    for label in labels:
        spaced = r"\s*".join(map(re.escape, label))
        m = re.search(spaced + r"[^\d\n]{0,12}?([\d,]{3,})\s*(천원|백만원|원)", text)
        if m:
            return int(m.group(1).replace(",", "")) * _UNIT[m.group(2)], label
    return _table_amount(text, labels)


_NUM_LINE = re.compile(r"^[\d,]{5,}$")
_TABLE_UNIT = re.compile(r"단위\s*[:：]?\s*(천원|백만원|원)")


def _table_amount(text: str, labels: list[str]) -> tuple[int | None, str | None]:
    """표 조판: 라벨 칸과 숫자 칸이 다른 줄로 나온다("도 급 액" / "835,174,000").

    단위는 표 머리의 "(단위:원)" 을 따른다. 단위 표기가 없으면 원으로 본다.
    변경계약 표는 당초·변경 순이라 첫 숫자(당초)를 쓴다 — 구간 판정엔 둘 다 같은 편이다.
    """
    lines = [re.sub(r"\s+", "", ln) for ln in text.splitlines() if ln.strip()]
    unit = _TABLE_UNIT.search(text)
    mult = _UNIT[unit.group(1)] if unit else 1
    for label in labels:
        for i, ln in enumerate(lines):
            if ln != label:
                continue
            for nxt in lines[i + 1 : i + 4]:
                if _NUM_LINE.match(nxt):
                    return int(nxt.replace(",", "")) * mult, f"{label}(표)"
    return None, None


class DelegationLookup:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        execution: dict[str, Any],
        embed: Callable[[list[str]], list[list[float]]] | None = None,
        vec_cache: Path | None = None,
    ) -> None:
        """``embed`` 가 있으면 BM25 와 벡터 순위를 RRF 로 합친다.

        벡터가 필요한 이유는 어휘 차이다. "조례 일부개정"과 별표의 "자치법규 제정·개정"은
        겹치는 글자가 없어 BM25 후보에 아예 안 올라온다(gold #13 #21).
        행 벡터는 ``vec_cache`` 에 저장해 두고 다시 쓴다.
        """
        self.rows = rows
        self.cfg = execution
        self.embed = embed
        self.vec_cache = vec_cache
        self._vecs: dict[int, list[float]] = {}
        if embed and vec_cache and vec_cache.expanduser().is_file():
            raw = json.loads(vec_cache.expanduser().read_text(encoding="utf-8"))
            self._vecs = {int(k): v for k, v in raw.items()}
        self._index: dict[str, tuple[list[dict[str, Any]], BM25Okapi]] = {}

    # --- 집행 품의 --------------------------------------------------------
    def is_execution(self, title: str) -> bool:
        flat = re.sub(r"\s+", "", title)
        has = lambda keys: any(re.sub(r"\s+", "", k) in flat for k in keys)  # noqa: E731
        if has(self.cfg.get("execution_strong", [])):
            return True
        suffix = self.cfg.get("not_execution_suffix")
        return has(self.cfg["execute_keywords"]) and not (suffix and re.search(suffix, flat))

    def execution(self, title: str, body: str) -> dict[str, Any]:
        category = next(
            c for c in self.cfg["categories"]
            if not c["match"] or any(m in title for m in c["match"])
        )
        amount, label = extract_labeled_amount(body, self.cfg["amount_labels"])
        out: dict[str, Any] = {
            "source": "amount_bracket", "category": category["key"],
            "amount": amount, "amount_label": label, "row": None, "level": None,
        }
        if amount is None:
            return out
        for r in self.rows:
            if not r["sheet"].startswith("(공통") or category["table"] not in r["item"]:
                continue
            lo, hi = r["amount_min"], r["amount_max"]
            if lo is None and hi is None:
                continue
            # 별표 표기: "초과"는 하한 배타, "이하"는 상한 포함
            if (lo is None or amount > lo) and (hi is None or amount <= hi):
                out.update(row=r["item"], level=r["final_level"])
                break
        return out

    # --- 사무명 검색 -------------------------------------------------------
    def candidates(self, dept: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        if dept not in self._index:
            pool = [
                r for r in self.rows
                if (r["sheet"] == dept or r["sheet"].startswith("(공통"))
                and r["amount_min"] is None and r["amount_max"] is None
            ]
            self._index[dept] = (pool, BM25Okapi([tokenize(_row_text(r)) for r in pool]))
        pool, index = self._index[dept]
        scores = index.get_scores(tokenize(query))
        bm_order = sorted(range(len(pool)), key=lambda i: -scores[i])
        if not self.embed:
            return [pool[i] | {"score": round(float(scores[i]), 2)} for i in bm_order[:k]]

        vecs = self._vectors(pool)
        q = _unit(self.embed([query])[0])
        sims = [sum(a * b for a, b in zip(q, v)) for v in vecs]
        vec_order = sorted(range(len(pool)), key=lambda i: -sims[i])
        fused: dict[int, float] = {}
        for order in (bm_order[: k * 4], vec_order[: k * 4]):
            for rank, i in enumerate(order):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [pool[i] | {"score": round(fused[i], 4)} for i in best]

    def _vectors(self, pool: list[dict[str, Any]]) -> list[list[float]]:
        missing = [r for r in pool if r["rid"] not in self._vecs]
        if missing:
            for r, v in zip(missing, self.embed([_row_text(r) for r in missing])):
                self._vecs[r["rid"]] = _unit(v)
            if self.vec_cache:
                self.vec_cache.expanduser().write_text(json.dumps(self._vecs), encoding="utf-8")
        return [self._vecs[r["rid"]] for r in pool]


def _row_text(r: dict[str, Any]) -> str:
    return f"{r['group']} {r['item']}".strip()


def _unit(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]
