import re
import unicodedata

from langchain_core.tools import BaseTool
from src.graph.state import ClinicalState
from src.graph.policies import is_prescription_request

class ClinicalNodes:
    def __init__(
        self,
        patient_tool: BaseTool,
        exams_tool: BaseTool,
        protocol_tool: BaseTool,
    ):
        self.patient_tool = patient_tool
        self.exams_tool = exams_tool
        self.protocol_tool = protocol_tool

    def validate(self, state: ClinicalState) -> dict:
        question = state.get("question", "").strip()

        if not question:
            return {
                "status": "invalid_query",
                "error": "A pergunta não pode estar vazia.",
            }

        # Identificação determinística de IDs sintéticos
        ids = set(re.findall(r"\bP\d+\b", question.upper()))
        provided_id = (state.get("patient_id") or "").strip().upper()

        if len(ids) > 1:
            return {
                "status": "invalid_query",
                "error": "Informe apenas um paciente por consulta.",
            }

        if provided_id and ids and provided_id not in ids:
            return {
                "status": "invalid_query",
                "error": "IDs de paciente conflitantes.",
            }

        patient_id = provided_id or (
            next(iter(ids)) if ids else None
        )

        if is_prescription_request(question):
            return {
                "question": question,
                "patient_id": patient_id,
                "status": "blocked_safety",
                "error": (
                    "O assistente não realiza prescrições, "
                    "indicações de medicamentos ou definições "
                    "de dosagem. Essas decisões exigem "
                    "avaliação de um profissional habilitado."
                ),
            }

        normalized_question = "".join(
            character
            for character in unicodedata.normalize(
                "NFKD", question.lower()
            )
            if not unicodedata.combining(character)
        )

        if patient_id:
            asks_about_pending_exams = (
                re.search(r"\bexames?\b", normalized_question)
                and re.search(r"\bpendentes?\b", normalized_question)
            )

            if not asks_about_pending_exams:
                return {
                    "question": question,
                    "patient_id": patient_id,
                    "status": "unsupported_intent",
                    "error": (
                        "Nesta versão, consultas específicas de "
                        "pacientes são limitadas à identificação "
                        "de exames pendentes."
                    ),
                }

        return {
            "question": question,
            "patient_id": patient_id,
            "status": "ready",
        }

    def fetch_patient(self, state: ClinicalState) -> dict:
        result = self.patient_tool.invoke({
            "patient_id": state["patient_id"]
        })

        if result["status"] != "ok":
            return {
                "status": (
                    "patient_not_found"
                    if result["status"] == "not_found"
                    else "unavailable"
                ),
                "error": result.get(
                    "message", "Falha na consulta do paciente."
                ),
            }

        return {"patient_result": result}

    def fetch_exams(self, state: ClinicalState) -> dict:
        result = self.exams_tool.invoke({
            "patient_id": state["patient_id"]
        })

        if result["status"] != "ok":
            return {
                "status": "unavailable",
                "error": "Não foi possível consultar os exames.",
            }

        return {"exams_result": result}

    def fetch_protocols(self, state: ClinicalState) -> dict:
        result = self.protocol_tool.invoke({
            "query": state["question"]
        })

        if result["status"] != "ok" or not result["matches"]:
            return {
                "status": (
                    "no_evidence"
                    if result["status"] == "no_results"
                    else "unavailable"
                ),
                "error": "Não foi possível recuperar protocolos.",
            }

        return {"protocol_result": result}

    def prepare_context(self, state: ClinicalState) -> dict:
        patient = state.get("patient_result", {}).get("patient")

        exams = state.get(
            "exams_result", {}
        ).get("pending_exams", [])

        protocols = state["protocol_result"]["matches"]

        sources = sorted({
            match["protocol_id"]
            for match in protocols
        })

        return {
            "result": {
                "status": "context_ready",
                "question": state["question"],
                "patient": patient,
                "pending_exams": exams,
                "protocols": protocols,
                "sources": sources,
            }
        }

    def failure(self, state: ClinicalState) -> dict:
        return {
            "result": {
                "status": state.get("status", "error"),
                "message": state.get(
                    "error", "Não foi possível concluir a consulta."
                ),
            }
        }