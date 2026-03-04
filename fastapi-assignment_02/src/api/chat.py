"""채팅 API 라우터.

이 파일의 목적:
- `/api/v1/chat` HTTP 엔드포인트를 제공합니다.

포함 내용:
- POST `/api/v1/chat` 라우트

사용 시점:
- 클라이언트가 채팅 요청을 보낼 때 사용합니다.
"""

from fastapi import APIRouter

from src.models.chat import ChatRequest
from src.services.chat_service import (
    generate_chat_response,
    generate_streaming_response,
)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=None)
def create_chat(request: ChatRequest):
    """채팅 요청을 받아 streaming/non-streaming 형식으로 응답합니다."""
    # Step 1. streaming 모드일 때는 SSE 형태로 데이터를 전송합니다.
    if request.streaming:
        return generate_streaming_response(request)

    # Step 2. non-streaming 모드일 때는 JSON 응답을 반환합니다.
    return generate_chat_response(request)
