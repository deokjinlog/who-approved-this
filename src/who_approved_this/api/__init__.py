"""데모용 HTTP API.

로컬에서 결과를 눈으로 보기 위한 얇은 껍데기다. **제품이 아니다** —
인증이 없고 localhost 전용이며, 계산은 전부 ``tracks/`` 코드를 그대로 호출한다.

계층
    ``models``    요청·응답 스키마(pydantic)
    ``services``  트랙 코드를 불러 결과를 만드는 곳. 로직을 여기서 새로 쓰지 않는다.
    ``routers``   HTTP 경로만. 서비스를 호출해 스키마로 감싼다.
"""

from who_approved_this.api.app import create_app

__all__ = ["create_app"]
