"""FastAPI 앱 조립.

인증이 없다. **localhost 전용 데모**로만 띄운다.
"""

from __future__ import annotations

from fastapi import FastAPI

from who_approved_this.api.routers import router


def create_app() -> FastAPI:
    """앱을 만든다. Swagger 는 ``/docs`` 가 아니라 ``/swagger`` 에 둔다.

    ``/docs`` 는 문서 목록 API 경로로 이미 쓰고 있어 충돌하기 때문이다.
    """
    app = FastAPI(
        title="who-approved-this (데모)",
        description=(
            "결재선 예측·품의서 초안 생성 리허설을 눈으로 보기 위한 로컬 API. "
            "인증 없음, localhost 전용, 제품 아님."
        ),
        version="0.1.0",
        docs_url="/swagger",
        redoc_url=None,
    )
    app.include_router(router)

    @app.on_event("startup")
    def _startup() -> None:
        """문서·색인·모델을 미리 깨워 첫 요청 지연을 없앤다."""
        from who_approved_this.api import services

        services.warm_up()

    return app


app = create_app()
