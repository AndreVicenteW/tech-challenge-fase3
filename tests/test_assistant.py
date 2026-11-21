from uuid import uuid4

import pytest
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graph.workflow import build_workflow


class FakeProvider:
    def __init__(self):
        self.calls = 0
        self.should_fail = False
        self.draft_override = None

    def generate_draft(self, question: str, context: str) -> str:
        self.calls += 1

        if self.should_fail:
            raise RuntimeError("Falha simulada da LLM")

        if self.draft_override is not None:
            return self.draft_override

        return (
            "Os exames pendentes para P001 são: "
            "Hemoglobina glicada."
        )


@pytest.fixture
def assistant():
    @tool
    def get_patient_info(patient_id: str) -> dict:
        """Consulta um paciente fictício."""
        if patient_id == "P999":
            return {
                "status": "not_found",
                "message": "Paciente não encontrado.",
            }

        return {
            "status": "ok",
            "patient": {
                "patient_id": patient_id,
                "age": 58,
                "conditions": ["Diabetes mellitus tipo 2"],
            },
        }

    @tool
    def get_pending_exams(patient_id: str) -> dict:
        """Consulta exames pendentes fictícios."""
        return {
            "status": "ok",
            "patient_id": patient_id,
            "pending_exams": [
                {
                    "exam_id": "E001",
                    "name": "Hemoglobina glicada",
                    "status": "pendente",
                }
            ],
            "count": 1,
        }

    @tool
    def search_protocols(query: str) -> dict:
        """Consulta protocolos fictícios."""
        return {
            "status": "ok",
            "matches": [
                {
                    "protocol_id": "PRT-001",
                    "source": "data/protocols/PRT-001.md",
                    "content": "Verificação de exames pendentes.",
                    "distance": 0.75,
                }
            ],
        }

    provider = FakeProvider()

    graph = build_workflow(
        patient_tool=get_patient_info,
        exams_tool=get_pending_exams,
        protocol_tool=search_protocols,
        llm_provider=provider,
        checkpointer=InMemorySaver(),
    )

    config = {
        "configurable": {
            "thread_id": str(uuid4())
        }
    }

    return graph, provider, config


def start_assistant(graph, config):
    return graph.invoke(
        {"question": "Quais exames estão pendentes para P001?"},
        config=config,
    )


def test_requires_human_approval(assistant):
    graph, provider, config = assistant

    state = start_assistant(graph, config)

    assert "__interrupt__" in state
    assert provider.calls == 1

    review = state["__interrupt__"][0].value

    assert review["type"] == "human_review"
    assert review["verified_exams"][0]["exam_id"] == "E001"
    assert "Hemoglobina glicada" in review["draft"]

    # Ainda não existe resposta aprovada
    assert state["result"]["status"] == "context_ready"


def test_approved_response(assistant):
    graph, provider, config = assistant

    start_assistant(graph, config)

    state = graph.invoke(
        Command(resume={"approved": True}),
        config=config,
    )

    result = state["result"]

    assert result["status"] == "approved"
    assert "Hemoglobina glicada" in result["answer"]
    assert result["verified_exams"][0]["exam_id"] == "E001"

    # A retomada não deve gerar o texto novamente
    assert provider.calls == 1


def test_rejected_response(assistant):
    graph, _, config = assistant

    start_assistant(graph, config)

    state = graph.invoke(
        Command(resume={"approved": False}),
        config=config,
    )

    assert state["result"]["status"] == "rejected"
    assert "answer" not in state["result"]


def test_invalid_approval_is_rejected(assistant):
    graph, _, config = assistant

    start_assistant(graph, config)

    state = graph.invoke(
        Command(resume={"approved": "sim"}),
        config=config,
    )

    assert state["result"]["status"] == "rejected"


def test_generation_failure(assistant):
    graph, provider, config = assistant

    provider.should_fail = True

    state = start_assistant(graph, config)

    assert "__interrupt__" not in state
    assert state["result"]["status"] == "generation_failed"
    assert provider.calls == 1


def test_unknown_patient_does_not_call_llm(assistant):
    graph, provider, config = assistant

    state = graph.invoke(
        {"question": "Quais exames estão pendentes para P999?"},
        config=config,
    )

    assert state["result"]["status"] == "patient_not_found"
    assert "__interrupt__" not in state
    assert provider.calls == 0

def test_hallucinated_exam_is_not_published(assistant):
    graph, provider, config = assistant

    # Simular uma alucinação do Qwen
    provider.draft_override = (
        "Os exames pendentes para P001 são: "
        "Tomografia e Exame de Urina, "
        "solicitados em 10/11/2026."
    )

    state = start_assistant(graph, config)

    assert "__interrupt__" in state

    review = state["__interrupt__"][0].value

    # O rascunho ainda contém a alucinação
    assert "Tomografia" in review["draft"]

    # A resposta proposta não contém exames inventados
    assert "Tomografia" not in review["proposed_answer"]
    assert "Exame de Urina" not in review["proposed_answer"]

    assert (
        "Hemoglobina glicada"
        in review["proposed_answer"]
    )

    assert (
        review["validation"]["mode"]
        == "structured_patient_data"
    )

    # Mesmo aprovando, a alucinação não será publicada
    final_state = graph.invoke(
        Command(resume={"approved": True}),
        config=config,
    )

    answer = final_state["result"]["answer"]

    assert "Hemoglobina glicada" in answer
    assert "Tomografia" not in answer
    assert "Exame de Urina" not in answer
    assert "10/11/2026" not in answer

def test_unsupported_patient_request(assistant):
    graph, provider, config = assistant

    state = graph.invoke(
        {
            "question": (
                "Qual é a idade do paciente P001?"
            )
        },
        config=config,
    )

    assert state["result"]["status"] == "unsupported_intent"
    assert "__interrupt__" not in state
    assert provider.calls == 0

@pytest.mark.parametrize(
    "question",
    [
        "Prescreva medicamentos para diabetes tipo 2.",
        "Qual medicamento devo prescrever para P001?",
        "Qual dose devo administrar?",
    ],
)
def test_prescription_is_blocked(
    assistant,
    question,
):
    graph, provider, config = assistant

    state = graph.invoke(
        {"question": question},
        config=config,
    )

    assert state["result"]["status"] == "blocked_safety"
    assert "__interrupt__" not in state
    assert provider.calls == 0