import logging

from src.audit.store import AuditStore
from src.graph.state import ClinicalState


logger = logging.getLogger(__name__)


class AuditNodes:
    def __init__(self, store: AuditStore):
        self.store = store

    def record(self, state: ClinicalState) -> dict:
        request_id = state.get("request_id")

        if not request_id:
            return {
                "audit_recorded": False,
                "result": {
                    "status": "audit_failed",
                    "message": "Identificador de execução ausente.",
                },
            }

        result = state["result"]

        # Em caso de falha, recupera as fontes do
        # estado intermediário, quando disponíveis.
        matches = state.get(
            "protocol_result", {}
        ).get("matches", [])

        sources = result.get("sources") or sorted({
            match["protocol_id"]
            for match in matches
            if match.get("protocol_id")
        })

        event = {
            "request_id": request_id,
            "question": state.get("question", ""),
            "patient_id": state.get("patient_id"),
            "status": result["status"],
            "sources": sources,
            "validation": state.get("validation"),
            "draft": state.get("draft"),
            "proposed_answer": state.get("proposed_answer"),
            "answer": result.get("answer"),
            "message": result.get("message"),
        }

        try:
            self.store.save(event)

        except Exception:
            logger.exception(
                "Falha ao registrar auditoria da execução."
            )

            return {
                "audit_recorded": False,
                "result": {
                    "status": "audit_failed",
                    "message": (
                        "Não foi possível registrar a auditoria. "
                        "A resposta não será liberada."
                    ),
                },
            }

        return {"audit_recorded": True}