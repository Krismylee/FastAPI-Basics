"""FastAPI 앱 진입점.

이 파일의 목적:
- 교육용 FastAPI 앱을 생성하고 라우터를 등록합니다.

포함 내용:
- FastAPI 앱 객체
- 실행용 main 함수

사용 시점:
- 서버 실행(`uv run uvicorn src.main:app --reload`) 또는 테스트 import 시 사용합니다.
"""

from fastapi import FastAPI
import uvicorn

from src.api.chat import router as chat_router

app = FastAPI(title="신입 개발자 FastAPI 튜토리얼", version="1.0.0")
app.include_router(chat_router)


def main() -> None:
    """로컬 개발 서버를 실행합니다."""
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
