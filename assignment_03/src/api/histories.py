"""채팅 히스토리 CRUD API 라우터.

이 파일의 목적:
- 채팅 히스토리 저장/조회/수정/삭제 엔드포인트를 제공합니다.

포함 내용:
- `/api/v1/chat-histories` 라우터

사용 시점:
- 테스트 또는 운영 도구에서 히스토리를 직접 확인/관리할 때 사용합니다.
"""

from fastapi import APIRouter, Depends, Query, status

from src.core.dependencies import get_chat_history_service
from src.models.history import (
    ChatHistoryCreateRequest,
    ChatHistoryEnvelope,
    ChatHistoryListEnvelope,
    ChatHistoryUpdateRequest,
    DeleteAllHistoryEnvelope,
    DeleteHistoryEnvelope,
)
from src.services.chat_history_service import ChatHistoryService

router = APIRouter(prefix="/api/v1/chat-histories", tags=["chat-histories"])


@router.post("", response_model=ChatHistoryEnvelope, status_code=status.HTTP_201_CREATED)
def create_history(
    request: ChatHistoryCreateRequest,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatHistoryEnvelope:
    """히스토리를 생성합니다."""
    history = service.create_history(request=request)
    return ChatHistoryEnvelope(data=history)


@router.get("", response_model=ChatHistoryListEnvelope)
def list_histories(
    session_id: int = Query(...),
    lt_query_id: int | None = Query(default=None),
    limit: int = Query(default=5),
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatHistoryListEnvelope:
    """히스토리 목록을 조회합니다."""
    # Step 1. service.list_histories를 호출하세요.
    history = service.list_histories(session_id=session_id, lt_query_id=lt_query_id, limit=limit)
    # Step 2. 반환값을 ChatHistoryListEnvelope로 감싸 반환하세요.
    return ChatHistoryListEnvelope(data=history)
    #raise NotImplementedError("list_histories API를 구현하세요.")


@router.get("/{session_id}/{query_id}", response_model=ChatHistoryEnvelope)
def get_history(
    session_id: int,
    query_id: int,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatHistoryEnvelope:
    """히스토리 단건을 조회합니다."""
    history = service.get_history(session_id=session_id, query_id=query_id)
    return ChatHistoryEnvelope(data=history)


@router.patch("/{session_id}/{query_id}", response_model=ChatHistoryEnvelope)
def update_history(
    session_id: int,
    query_id: int,
    request: ChatHistoryUpdateRequest,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatHistoryEnvelope:
    """히스토리를 수정합니다."""
    history = service.update_history(session_id=session_id, query_id=query_id, request=request)
    return ChatHistoryEnvelope(data=history)


@router.delete("/{session_id}/{query_id}", response_model=DeleteHistoryEnvelope)
def delete_history(
    session_id: int,
    query_id: int,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> DeleteHistoryEnvelope:
    """히스토리를 삭제합니다."""
    service.delete_history(session_id=session_id, query_id=query_id)
    return DeleteHistoryEnvelope(data={"session_id": session_id, "query_id": query_id})


@router.delete("", response_model=DeleteAllHistoryEnvelope)
def delete_all_histories(
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> DeleteAllHistoryEnvelope:
    """히스토리 전체를 삭제합니다."""
    # Step 1. service.delete_all_histories를 호출하세요.
    deleted_count = service.delete_all_histories()
    # Step 2. deleted_count를 응답 모델에 넣어 반환하세요.
    return DeleteAllHistoryEnvelope(data={"deleted_count": deleted_count})
    #raise NotImplementedError("delete_all_histories API를 구현하세요.")

