"""채팅 히스토리 CRUD 유스케이스 서비스.

이 파일의 목적:
- 히스토리 저장소 접근을 API 관점의 유스케이스로 조율합니다.

포함 내용:
- ChatHistoryService 클래스

사용 시점:
- 라우터와 채팅 완성 서비스에서 히스토리 로직이 필요할 때 사용합니다.
"""

import sqlite3

from src.core.errors import BadRequestError, ConflictError, NotFoundError
from src.models.history import (
    ChatHistoryCreateRequest,
    ChatHistoryRead,
    ChatHistoryUpdateRequest,
)
from src.repositories.chat_history_repository import ChatHistoryRepository


class ChatHistoryService:
    """채팅 히스토리 유스케이스 서비스."""

    def __init__(self, repository: ChatHistoryRepository) -> None:
        self._repository = repository

    def create_history(self, request: ChatHistoryCreateRequest) -> ChatHistoryRead:
        """히스토리를 생성합니다."""
        # Step 1. repository.create를 호출하세요.
        try:
            history = self._repository.create(
                session_id=request.session_id,
                query_id=request.query_id,
                user_message=request.user_message,
                intent=request.intent,
                final_answer=request.final_answer
            )
            return history
        # Step 2. sqlite3.IntegrityError가 발생하면 ConflictError로 변환하세요.
        except sqlite3.IntegrityError:
            raise ConflictError(
                f"History already exists: "
                f"session_id={request.session_id}, query_id={request.query_id}"
            )
        #raise NotImplementedError("create_history를 구현하세요.")

    def list_histories(
        self,
        session_id: int,
        lt_query_id: int | None = None,
        limit: int = 5,
    ) -> list[ChatHistoryRead]:
        """세션 히스토리를 조회합니다."""
        # Step 1. limit가 1 이상인지 검증하세요(아니면 BadRequestError).
        if limit < 1:
            raise BadRequestError("Limit must be >= 1")
        # Step 2. lt_query_id가 없으면 list_by_session, 있으면 list_before_query_id를 호출하세요.
        if lt_query_id is None:
            histories = self._repository.list_by_session(
                session_id=session_id,
                limit=limit
            )
        else:
            histories = self._repository.list_before_query_id(
                session_id=session_id,
                query_id=lt_query_id,
                limit=limit
            )
        return histories
        #raise NotImplementedError("list_histories를 구현하세요.")

    def get_history(self, session_id: int, query_id: int) -> ChatHistoryRead:
        """히스토리 단건을 조회합니다."""
        history = self._repository.get(session_id=session_id, query_id=query_id)
        if history is None:
            raise NotFoundError("채팅 히스토리", f"{session_id}:{query_id}")
        return history

    def update_history(
        self,
        session_id: int,
        query_id: int,
        request: ChatHistoryUpdateRequest,
    ) -> ChatHistoryRead:
        """히스토리를 수정합니다."""
        history = self._repository.update(
            session_id=session_id,
            query_id=query_id,
            user_message=request.user_message,
            intent=request.intent,
            final_answer=request.final_answer,
        )
        if history is None:
            raise NotFoundError("채팅 히스토리", f"{session_id}:{query_id}")
        return history

    def delete_history(self, session_id: int, query_id: int) -> None:
        """히스토리를 삭제합니다."""
        deleted = self._repository.delete(session_id=session_id, query_id=query_id)
        if not deleted:
            raise NotFoundError("채팅 히스토리", f"{session_id}:{query_id}")

    def delete_all_histories(self) -> int:
        """히스토리 전체를 삭제합니다."""
        # Step 1. repository.delete_all을 호출하세요.
        deleted_count = self._repository.delete_all()
        # Step 2. 삭제된 건수를 반환하세요.
        return deleted_count
        #raise NotImplementedError("delete_all_histories를 구현하세요.")

