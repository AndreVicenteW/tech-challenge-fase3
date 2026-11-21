from langchain_core.documents import Document

from src.tools.protocol_tools import create_protocol_tool


class FakeVectorStore:
    def similarity_search_with_score(self, query, k):
        return [
            (
                Document(
                    page_content=(
                        "Exames pendentes devem ser "
                        "identificados nos registros."
                    ),
                    metadata={
                        "protocol_id": "PRT-001",
                        "source": "data/protocols/PRT-001.md",
                    },
                ),
                0.75,
            )
        ][:k]


class EmptyVectorStore:
    def similarity_search_with_score(self, query, k):
        return []


class FailingVectorStore:
    def similarity_search_with_score(self, query, k):
        raise RuntimeError("Database unavailable")


def test_search_protocols():
    search_tool = create_protocol_tool(
        vector_store=FakeVectorStore()
    )

    result = search_tool.invoke({
        "query": "Quais exames estão pendentes?"
    })

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["matches"][0]["protocol_id"] == "PRT-001"
    assert "Exames pendentes" in result["matches"][0]["content"]


def test_empty_query():
    search_tool = create_protocol_tool(
        vector_store=FakeVectorStore()
    )

    result = search_tool.invoke({"query": "   "})

    assert result["status"] == "invalid_query"


def test_no_results():
    search_tool = create_protocol_tool(
        vector_store=EmptyVectorStore()
    )

    result = search_tool.invoke({
        "query": "Protocolo desconhecido"
    })

    assert result["status"] == "no_results"
    assert result["matches"] == []


def test_unavailable_store():
    search_tool = create_protocol_tool(
        vector_store=FailingVectorStore()
    )

    result = search_tool.invoke({
        "query": "Exames pendentes"
    })

    assert result["status"] == "unavailable"
    assert result["matches"] == []