"""T1c 결재선 예측기 — 같은 인터페이스를 가진 세 가지 방식.

``predict(refs, target) -> [{"role", "title", "name"}]`` 하나만 맞추면 된다.

트랙 안에 두는 이유
    입력이 ``pdf_path`` 가 아니라 ``(참고 문서들, 대상 문서)`` 라서
    :class:`~who_approved_this.parse.base.Parser` 계약에 맞지 않는다.
    골격(base 인터페이스·패키지 구조)을 건드리지 않기로 했으므로,
    트랙 전용 모듈로 둔다.

방식
    ``baseline_copy``  같은 부서 직전 문서 결재선을 그대로 복사
    ``baseline_vote``  칸 위치별로 참고 문서 다수결(동률이면 최근 것)
    ``llm_local``      ollama 로컬 모델에 참고 문서와 대상 문서를 주고 JSON 요청
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Protocol

#: 프롬프트에 넣을 본문 길이. 결재란 자체가 아니라 "무슨 문서인가"만 알면 된다.
BODY_CHARS = 1500

ROLES = ("기안", "검토", "결재", "협조")


class Predictor(Protocol):
    """결재선 예측기 공통 계약."""

    name: str

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        """참고 문서들을 보고 ``target`` 의 결재선을 예측한다."""
        ...


def _recent_first(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """생산일자 내림차순(같으면 문서번호 내림차순)으로 정렬한다."""
    return sorted(
        refs, key=lambda r: (r.get("생산일자", ""), r.get("문서번호", "")), reverse=True
    )


class BaselineCopy:
    """같은 부서 **직전 문서**의 결재선을 그대로 쓴다.

    "부서가 같으면 결재선도 같다"는 가장 단순한 가정. 이 가정이 어디서 깨지는지
    보는 게 목적이므로, 잘 맞든 안 맞든 그 자체가 결과다.
    """

    name = "baseline_copy"

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        if not refs:
            return []
        latest = _recent_first(refs)[0]
        return [dict(c) for c in latest["cells"]]


class BaselineVote:
    """칸 위치별로 참고 문서 **다수결**. 동률이면 최근 문서를 따른다."""

    name = "baseline_vote"

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        if not refs:
            return []
        ordered = _recent_first(refs)
        out: list[dict[str, str]] = []
        for role in ROLES:
            per_ref = [[c for c in r["cells"] if c["role"] == role] for r in ordered]
            n_cells = Counter(len(c) for c in per_ref).most_common(1)[0][0]
            for i in range(n_cells):
                cands = [cells[i] for cells in per_ref if i < len(cells)]
                if not cands:
                    continue
                out.append(
                    {
                        "role": role,
                        "title": _vote([c["title"] for c in cands]),
                        "name": _vote([c["name"] for c in cands]),
                    }
                )
        return out


def _vote(values: list[str]) -> str:
    """다수결. 동률이면 목록 앞쪽(=더 최근 문서)을 따른다."""
    counts = Counter(values)
    top = max(counts.values())
    for v in values:  # values 는 최근 순
        if counts[v] == top:
            return v
    return values[0]


class VoteTitlesLLMNames:
    """직위는 다수결, 이름은 LLM — 각자 잘하는 쪽만 쓴다.

    12케이스 실측에서 baseline_vote 는 **직위**(66.2% vs 48.2%)와 칸 수에서 이기고,
    LLM 은 **이름**(64.2% vs 42.1%)에서 크게 이겼다. 이유도 갈렸다.
    다수결은 표기를 정확히 복사하되 사람을 못 고르고, LLM 은 문서 내용을 읽어
    사람을 고르되 직위 칸에 팀 이름을 적는다.

    그래서 칸 구조와 직위는 다수결이 만든 것을 쓰고, 같은 자리의 이름만 LLM 것으로
    바꾼다. LLM 이 그 자리에 답을 못 내면 다수결 이름을 그대로 둔다.
    """

    def __init__(self, model: str, timeout: int = 600) -> None:
        self.vote = BaselineVote()
        self.llm = LocalLLM(model, timeout)
        self.name = f"vote_titles+llm_names({model})"

    @property
    def retries(self) -> int:
        return self.llm.retries

    @property
    def failures(self) -> int:
        return self.llm.failures

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        skeleton = self.vote.predict(refs, target)
        names = self.llm.predict(refs, target)
        # 역할별로 순서를 맞춰 이름만 옮긴다.
        by_role: dict[str, list[str]] = {}
        for cell in names:
            by_role.setdefault(cell["role"], []).append(cell["name"])
        used: dict[str, int] = {}
        out: list[dict[str, str]] = []
        for cell in skeleton:
            i = used.get(cell["role"], 0)
            used[cell["role"]] = i + 1
            pool = by_role.get(cell["role"], [])
            name = pool[i] if i < len(pool) and pool[i] else cell["name"]
            out.append({**cell, "name": name, "evidence": "vote+llm"})

        # 다수결이 **한 칸도 못 만든 역할**은 LLM 제안을 그대로 쓴다.
        # 협조가 그렇다 — 조직 8건 중 1건에만 협조가 있으면 다수결은 항상 0칸을 내고,
        # 실제로 협조가 있는 문서에서 그 행이 통째로 비었다(E2E D-6).
        # 협조 유무는 문서 성격이 정하는 것이라(지출→경리 협조) 내용을 읽는 쪽이 낫다.
        covered = {c["role"] for c in skeleton}
        for role in ROLES:
            if role in covered:
                continue
            for cell in (c for c in names if c["role"] == role):
                out.append({**cell, "evidence": "llm-only"})
        return out


class LocalLLM:
    """ollama 로컬 모델에 결재선을 물어본다.

    문서 내용(제목·본문)과 참고 결재선이 프롬프트에 들어가므로 **실명이 오간다.**
    ollama 는 로컬에서 돌고 외부로 나가지 않는다. 프롬프트 자체는 저장하지 않는다.
    """

    def __init__(self, model: str, timeout: int = 600) -> None:
        self.model = model
        self.timeout = timeout
        self.name = f"llm_local({model})"
        #: 마지막 호출의 실패 사유 — 진단용. 값은 저장하지 않는다.
        self.last_error = ""
        #: JSON 파싱에 실패해 재시도한 횟수 / 재시도까지 실패한 횟수.
        self.retries = 0
        self.failures = 0

    def predict(
        self, refs: list[dict[str, Any]], target: dict[str, Any]
    ) -> list[dict[str, str]]:
        """한 번 물어보고, JSON 파싱에 실패하면 **한 번만** 다시 물어본다.

        두 번째도 실패하면 빈 목록을 돌려주고 :attr:`failures` 에 센다.
        빈 결과는 "칸 수 0"으로 채점되므로 실패가 점수에 그대로 드러난다.
        """
        prompt = build_prompt(refs, target, system_in_prompt=False)
        for attempt in (1, 2):
            self.last_error = ""
            raw = ollama_chat(
                self.model, SYSTEM, prompt, temperature=0.0, timeout=self.timeout
            )
            if raw is None:
                self.last_error = "ollama 호출 실패"
            else:
                cells = parse_cells(raw, self)
                if cells:
                    return cells
            if attempt == 1:
                self.retries += 1
        self.failures += 1
        return []


#: Qwen3 계열은 기본이 thinking 모드라 답 앞에 사고 과정이 붙는다.
#: ollama 가 ``think`` 를 지원하면 끄고, 지원하지 않으면 프롬프트 지시로 대신한다.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def ollama_generate(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = 300,
    think: bool = False,
) -> str | None:
    """ollama ``/api/generate`` 로 한 번 생성한다. 실패하면 ``None``.

    CLI(``ollama run``)로는 temperature 를 줄 수 없어 HTTP API 를 쓴다.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if not think:
        payload["think"] = False
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("response", "")
    except urllib.error.HTTPError:
        # think 파라미터를 모르는 구버전이면 빼고 한 번 더.
        if not think:
            payload.pop("think", None)
            try:
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read()).get("response", "")
            except (urllib.error.URLError, OSError, TimeoutError):
                return None
        return None
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def ollama_chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    timeout: int = 600,
    think: bool = False,
    num_ctx: int | None = None,
) -> str | None:
    """ollama ``/api/chat`` 으로 한 번 대화한다. 실패하면 ``None``.

    ``num_ctx`` — 긴 원문(여러 쪽 통째 요약)은 기본 문맥 창을 넘으면 **조용히 잘린다.** 그때만 키운다.

    ``/api/generate`` 대신 이걸 쓴다. generate 는 **이어쓰기** 엔드포인트라
    프롬프트가 문서 모양이면 모델이 그 문서를 계속 이어 쓴다 — 실제로 초안에
    ``## 참고문서 2``, ``본문:`` 같은 내부 마커와 참고 문서 본문이 그대로 복사돼 나왔다.
    chat 은 system/user 역할이 분리돼 지시와 자료가 섞이지 않는다.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, **({"num_ctx": num_ctx} if num_ctx else {})},
    }
    if not think:
        payload["think"] = False

    def _post(body: dict[str, Any]) -> str | None:
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("message", {}).get("content", "")

    try:
        return _post(payload)
    except urllib.error.HTTPError:
        if not think:  # think 를 모르는 구버전이면 빼고 재시도
            payload.pop("think", None)
            try:
                return _post(payload)
            except (urllib.error.URLError, OSError, TimeoutError):
                return None
        return None
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


SYSTEM = """너는 한국 행정기관의 전자결재 시스템이다. 문서의 결재선(기안·검토·결재·협조)을 예측한다.

규칙:
- 같은 부서의 과거 문서 결재선을 우선 근거로 삼는다.
- 결재선은 아래에서 위로 올라간다. 첫 칸이 기안자(실무자), 마지막 칸이 최종 결재자다.
- 직위(title)는 과거 문서에 나온 표기를 그대로 쓴다. 새 직위를 만들지 않는다.
- 이름(name)도 과거 문서에 나온 사람 중에서 고른다. 모르면 빈 문자열로 둔다.
- 문서 내용이 과거 문서와 다른 성격이면 칸 수나 직위가 달라질 수 있다.
- **협조**는 다른 부서의 소관이 걸릴 때 붙는 칸이다. 지출·예산 집행 문서는 회계·경리
  담당이, 계약은 계약 담당이 협조로 들어간다. 과거 문서 중 성격이 같은 것에
  협조가 있으면 그 패턴을 따르고, 없으면 협조 칸을 만들지 않는다.

출력은 JSON 배열 하나만. 설명·코드블록 없이 배열만 출력한다.
[{"role": "기안|검토|결재|협조", "title": "직위", "name": "이름"}]"""


def build_prompt(
    refs: list[dict[str, Any]], target: dict[str, Any], system_in_prompt: bool = True
) -> str:
    """프롬프트를 만든다. 템플릿은 ``t1c_prompt.md`` 와 같은 구조다.

    ``system_in_prompt=False`` 면 지시문을 빼고 자료만 담는다(chat 의 user 메시지용).
    """
    blocks = []
    for i, ref in enumerate(_recent_first(refs), start=1):
        line = "\n".join(
            f"{c['role']} | {c['title']} | {c['name']}" for c in ref["cells"]
        )
        blocks.append(
            f"## 과거문서 {i}\n제목: {ref['제목']}\n본문:\n{ref['body'][:BODY_CHARS]}\n"
            f"결재선:\n{line}"
        )
    return (
        (f"{SYSTEM}\n\n" if system_in_prompt else "")
        + f"# 같은 부서의 과거 결재문서 {len(refs)}건\n\n"
        + "\n\n".join(blocks)
        + f"\n\n# 결재선을 예측할 문서\n\n제목: {target['제목']}\n본문:\n"
        f"{target['body'][:BODY_CHARS]}\n\n위 문서의 결재선을 JSON 배열로 출력하라."
    )


def _json_arrays(text: str) -> list[str]:
    """괄호 균형을 세어 완결된 ``[...]`` 덩어리를 모두 찾는다.

    사고 과정에도 대괄호가 섞이므로 "첫 [ 부터 마지막 ] 까지" 로 자르면 깨진다.
    """
    out: list[str] = []
    depth = start = 0
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start : i + 1])
    return out


def parse_cells(raw: str, owner: LocalLLM | None = None) -> list[dict[str, str]]:
    """모델 출력에서 JSON 배열을 건져 칸 목록으로 만든다.

    설명이 앞뒤로 붙거나 코드블록으로 감싸도, thinking 블록이 섞여도 배열만 뽑는다.
    후보가 여럿이면 **dict 를 담은 가장 긴 배열**을 답으로 본다. 실패하면 빈 목록.
    """
    text = _THINK_BLOCK.sub("", raw)
    best: Any = None
    for chunk in _json_arrays(text):
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and any(isinstance(x, dict) for x in parsed):
            if best is None or len(parsed) > len(best):
                best = parsed
    if best is None:
        if owner is not None:
            owner.last_error = "JSON 배열 없음"
        return []
    data = best
    out: list[dict[str, str]] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        out.append(
            {
                "role": role if role in ROLES else "검토",
                "title": str(item.get("title", "")).strip(),
                "name": str(item.get("name", "")).strip(),
            }
        )
    return out
