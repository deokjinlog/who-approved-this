"""리포 전역 설정.

데이터(원문 PDF·정답 텍스트·중간 산출물)는 리포 밖에 둔다. 기본 위치는
``~/data/who-approved-this`` 이고, 환경변수 ``WAT_DATA_ROOT`` 로 덮어쓴다.
"""

from __future__ import annotations

import os
from pathlib import Path

#: 데이터 루트. 리포 안에는 어떤 원문도 커밋하지 않는다.
DATA_ROOT: Path = Path(
    os.environ.get("WAT_DATA_ROOT", Path.home() / "data" / "who-approved-this")
).expanduser()

#: 결과표·점수 JSON을 쌓는 곳. 실명 등 개인정보는 남기지 않는다.
RESULTS_ROOT: Path = Path(__file__).resolve().parents[2] / "results"


def track_data_dir(track: str) -> Path:
    """트랙별 입력 데이터 디렉터리(``DATA_ROOT/<track>``)를 돌려준다."""
    return DATA_ROOT / track


def track_results_dir(track: str) -> Path:
    """트랙별 결과 디렉터리(``results/<track>``)를 돌려준다. 없으면 만든다."""
    d = RESULTS_ROOT / track
    d.mkdir(parents=True, exist_ok=True)
    return d
