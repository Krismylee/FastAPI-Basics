"""채팅 API 요청/응답 스키마.

이 파일의 목적:
- `/api/v1/chat` 엔드포인트의 요청/응답 모델을 정의합니다.

포함 내용:
- ChatRequest 모델 자리
- ChatResponse 모델 자리

사용 시점:
- 실습에서 Pydantic 모델을 직접 구현할 때 사용합니다.
"""

#raise NotImplementedError("TODO: src/models/chat.py의 ChatRequest/ChatResponse를 구현하세요.")
from uuid import UUID
from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: UUID
    query_id: int
    streaming: bool
    message: str
    
class ChatResponse(BaseModel):
    message: str