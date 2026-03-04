"""RAG 목업 문서 저장소.

이 파일의 목적:
- RAG 분기에서 사용할 고정 문서 3개를 제공합니다.

포함 내용:
- get_mock_rag_documents 함수

사용 시점:
- 실제 검색기 없이 분기 흐름과 응답 포맷을 검증할 때 사용합니다.
"""

from src.models.chat import RagDocument


def get_mock_rag_documents() -> list[RagDocument]:
    """LangGraph 주제의 고정 문서 3개를 반환합니다."""
    return [
        RagDocument(
            title="LangGraph 상태 모델 기초",
            content=(
                "LangGraph는 상태 딕셔너리를 중심으로 노드 간 데이터를 전달합니다. "
                "초기 상태 키를 명확히 두면 분기 흐름 추적이 쉬워집니다."
            ),
            page_number=0,
        ),
        RagDocument(
            title="조건 분기와 라우팅 패턴",
            content=(
                "의도 분류 노드 뒤에서 조건 분기를 연결하면 RAG 경로와 일반 생성 경로를 "
                "명확히 분리할 수 있습니다."
            ),
            page_number=0,
        ),
        RagDocument(
            title="스트리밍 응답 설계 팁",
            content=(
                "최종 생성 노드에서 스트리밍을 사용할 때는 chunk 이벤트와 final 이벤트를 "
                "분리해 전송하면 클라이언트 처리 로직이 단순해집니다."
            ),
            page_number=0,
        ),
    ]

