"""채팅 API 라우터.

이 파일의 목적:
- `/api/v1/chat` 엔드포인트 라우팅을 제공합니다.

포함 내용:
- POST `/api/v1/chat` 라우트 자리

사용 시점:
- 실습에서 FastAPI 라우터를 직접 구현할 때 사용합니다.
"""

#raise NotImplementedError("TODO: src/api/chat.py의 /api/v1/chat 라우터를 구현하세요.")

from fastapi import APIRouter
from src.models.chat import ChatRequest, ChatResponse
from src.services.chat_service import generate_chat_response

router = APIRouter(prefix="/api/v1", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return generate_chat_response(request)
