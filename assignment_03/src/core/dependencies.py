"""의존성 주입 구성.

파일의 목적:
- FastAPI Depends로 사용하는 객체 생성 함수를 중앙에서 관리한다.

포함 내용:
- 설정/저장소/서비스/게이트웨이 생성 의존성 함수

사용 시점:
- 라우터 함수에서 서비스 객체를 주입받아 사용할 때 사용한다.
"""

from fastapi import Depends, Request

from src.core.config import AppSettings, load_app_settings
from src.repositories.chat_history_repository import ChatHistoryRepository
from src.services.chat_completion_service import ChatCompletionService
from src.services.chat_history_service import ChatHistoryService
from src.services.llm_gateway import HttpLlmGateway, LlmGateway
from src.services.rag_graph.run import ChatGraphRunner


def get_settings(request: Request) -> AppSettings:
    """앱 상태 또는 환경 변수에서 설정을 읽어온다."""
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, AppSettings):
        return settings
    return load_app_settings()


def get_chat_history_repository(
    settings: AppSettings = Depends(get_settings),
) -> ChatHistoryRepository:
    """채팅 히스토리 저장소 객체를 생성한다."""
    return ChatHistoryRepository(database_path=settings.database_path)


def get_chat_history_service(
    repository: ChatHistoryRepository = Depends(get_chat_history_repository),
) -> ChatHistoryService:
    """채팅 히스토리 서비스 객체를 생성한다."""
    return ChatHistoryService(repository=repository)


def get_llm_gateway(
    settings: AppSettings = Depends(get_settings),
) -> LlmGateway:
    """외부 LLM 게이트웨이 객체를 생성한다."""
    return HttpLlmGateway(
        endpoint=settings.llm_endpoint,
        model=settings.llm_model,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def get_chat_graph_runner(
    llm_gateway: LlmGateway = Depends(get_llm_gateway),
) -> ChatGraphRunner:
    """분기 그래프 실행기를 생성한다."""
    return ChatGraphRunner(llm_gateway=llm_gateway)


def get_chat_completion_service(
    llm_gateway: LlmGateway = Depends(get_llm_gateway),
    history_service: ChatHistoryService = Depends(get_chat_history_service),
    graph_runner: ChatGraphRunner = Depends(get_chat_graph_runner),
) -> ChatCompletionService:
    """채팅 완성 서비스 객체를 생성한다."""
    return ChatCompletionService(
        llm_gateway=llm_gateway,
        history_service=history_service,
        graph_runner=graph_runner,
    )
