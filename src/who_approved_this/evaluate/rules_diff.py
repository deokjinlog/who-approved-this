"""귀납 규칙 vs 규정 규칙 대조 (골격).

**아직 구현하지 않았다.** 두 벌의 ``rules/approval.yaml`` 을 받아 무엇이 다른지
숫자로 낸다. 미팅에서 쓸 문장은 이것이다 —
**"전결규정 없이 이력만으로 가면 어디서, 얼마나 틀리는가."**

세 가지를 잰다.
    * **커버리지** — 규정에는 있는데 이력에 안 나타난 규칙(표본이 못 본 조건).
      금액 구간이 대표적이다. 이력에는 관측된 금액대만 나온다.
    * **충돌** — 같은 조건인데 결론이 다른 규칙(이력이 잘못 일반화한 것).
    * **과잉 일반화** — 이력이 만든 규칙 중 규정에 근거가 없는 것.
"""

from __future__ import annotations

from typing import Any


def diff_rules(induced: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    """두 규칙 뭉치를 대조한다.

    돌려주는 값(전부 수치·식별자. 실명은 담지 않는다)::

        {"organizations": {...},
         "delegation": {"only_official": n, "only_induced": n, "conflict": n},
         "cooperation": {...},
         "chain": {"match": bool, "missing_titles": [...]},
         "coverage_ratio": float,    # 규정 규칙 중 이력이 재현한 비율
         "conflict_ratio": float}
    """
    raise NotImplementedError("두 벌이 준비된 뒤 구현")


def score_predictions_by_source(
    cases: list[dict[str, Any]], evidence_field: str = "evidence"
) -> dict[str, Any]:
    """예측 결과를 **근거 태그별**로 갈라 정확도를 낸다.

    칸마다 ``roster | rule | vote | llm`` 태그가 붙으므로, 어느 층이 정확도를
    올리고 어느 층이 틀리는지 분리해서 볼 수 있다. ⑤ 단계의 핵심 산출물이다.
    """
    raise NotImplementedError("계층화 예측기 구현 후")


def delegation_impact(diff: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """규정과 이력이 다른 지점이 **실제 케이스에서 몇 건 틀리게 했는지** 센다.

    규칙 차이가 몇 개인지보다, 그 차이가 결재선 예측을 몇 건 망쳤는지가 중요하다.
    """
    raise NotImplementedError("두 벌이 준비된 뒤 구현")
