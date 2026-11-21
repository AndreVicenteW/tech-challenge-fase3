import json
import logging

from langgraph.types import interrupt

from src.graph.state import ClinicalState

logger = logging.getLogger(__name__)


class ResponseNodes:
    def __init__(self, provider):
        self.provider = provider

    def generate(self, state: ClinicalState) -> dict:
        result = state["result"]

        context = {
            "patient": result["patient"],
            "pending_exams": result["pending_exams"],
            "protocols": [
                {
                    "protocol_id": item["protocol_id"],
                    "source": item["source"],
                    "content": item["content"],
                }
                for item in result["protocols"]
            ],
        }

        try:
            draft = self.provider.generate_draft(
                question=state["question"],
                context=json.dumps(
                    context,
                    ensure_ascii=False,
                ),
            )

            if not draft.strip():
                raise ValueError("Resposta vazia.")

        except Exception:
            logger.exception("Falha na geração da resposta.")

            return {
                "status": "generation_failed",
                "error": "Não foi possível gerar a resposta.",
            }

        return {
            "draft": draft,
            "status": "draft_ready",
        }

    def human_review(self, state: ClinicalState) -> dict:
        context = state["result"]

        decision = interrupt({
            "type": "human_review",
            "question": state["question"],
            "draft": state["draft"],
            "proposed_answer": state["proposed_answer"],
            "validation": state["validation"],
            "verified_patient": context["patient"],
            "verified_exams": context["pending_exams"],
            "sources": context["sources"],
        })

        if (
            not isinstance(decision, dict)
            or decision.get("approved") is not True
        ):
            return {
                "result": {
                    "status": "rejected",
                    "message": "Resposta não aprovada.",
                }
            }

        return {
            "result": {
                "status": "approved",
                "answer": state["proposed_answer"],
                "verified_patient": context["patient"],
                "verified_exams": context["pending_exams"],
                "sources": context["sources"],
                "validation": state["validation"],
            }
        }

    def validate_response(self, state: ClinicalState) -> dict:
        context = state["result"]
        patient = context.get("patient")

        # Perguntas gerais: ainda não temos um verificador
        # automático de correção clínica.
        if patient is None:
            return {
                "proposed_answer": state["draft"],
                "validation": {
                    "status": "manual_review_required",
                    "mode": "llm_draft",
                    "message": (
                        "O conteúdo não foi validado automaticamente. "
                        "Exige revisão humana."
                    ),
                },
                "status": "validation_ready",
            }

        # Para perguntas sobre pacientes, os dados devem
        # corresponder ao registro consultado no SQLite.
        patient_id = patient.get("patient_id")

        if patient_id != state.get("patient_id"):
            return {
                "status": "validation_failed",
                "error": "Identificação do paciente inconsistente.",
            }

        exams = context.get("pending_exams")

        if not isinstance(exams, list):
            return {
                "status": "validation_failed",
                "error": "Formato inválido dos exames.",
            }

        for exam in exams:
            if (
                not isinstance(exam, dict)
                or not exam.get("exam_id")
                or not exam.get("name")
                or exam.get("status") != "pendente"
            ):
                return {
                    "status": "validation_failed",
                    "error": "Dados de exames inconsistentes.",
                }

        # Resposta construída exclusivamente com dados
        # estruturados, sem reutilizar o texto da LLM.
        if exams:
            exam_names = "; ".join(
                f"{exam['name']} ({exam['exam_id']})"
                for exam in exams
            )

            answer = (
                f"Segundo os registros do hospital, "
                f"os exames pendentes para o paciente "
                f"{patient_id} são: {exam_names}."
            )
        else:
            answer = (
                f"Segundo os registros do hospital, "
                f"não há exames pendentes cadastrados "
                f"para o paciente {patient_id}."
            )

        return {
            "proposed_answer": answer,
            "validation": {
                "status": "structural_pass",
                "mode": "structured_patient_data",
                "source": "SQLite",
                "message": (
                    "Resposta construída com os registros "
                    "estruturados, independentemente do rascunho."
                ),
            },
            "status": "validation_ready",
        }