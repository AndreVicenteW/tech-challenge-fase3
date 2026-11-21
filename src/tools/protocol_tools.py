import logging
from typing import Any

from langchain_core.tools import BaseTool, tool

from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def create_protocol_tool(
    vector_store: Any = None,
    k: int = 4,
) -> BaseTool:

    if k < 1:
        raise ValueError("k deve ser maior que zero.")

    store = (
        vector_store
        if vector_store is not None
        else get_vector_store()
    )

    @tool
    def search_protocols(query: str) -> dict:
        """Busca trechos relevantes nos protocolos internos do Hospital Aurora."""

        query = query.strip()

        if not query:
            return {
                "status": "invalid_query",
                "message": "A pergunta não pode estar vazia.",
                "matches": [],
            }

        try:
            results = store.similarity_search_with_score(
                query=query,
                k=k,
            )

        except Exception:
            logger.exception("Falha na consulta aos protocolos.")

            return {
                "status": "unavailable",
                "message": "Não foi possível consultar os protocolos.",
                "matches": [],
            }

        matches = []

        for document, distance in results:
            if not document.page_content.strip():
                continue

            protocol_id = document.metadata.get("protocol_id")

            if not protocol_id:
                continue

            matches.append({
                "protocol_id": protocol_id,
                "source": document.metadata.get("source"),
                "content": document.page_content,
                "distance": float(distance),
            })

        return {
            "status": "ok" if matches else "no_results",
            "query": query,
            "count": len(matches),
            "matches": matches,
        }

    return search_protocols