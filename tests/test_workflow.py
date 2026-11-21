import pytest

from langchain_core.tools import tool
from src.graph.workflow import build_workflow


@pytest.fixture
def graph_with_tools():
    calls = {
        "patient": [],
        "exams": [],
        "protocols": [],
    }

    @tool
    def get_patient_info(patient_id: str) -> dict:
        """Consulta um paciente fictício pelo ID."""
        calls["patient"].append(patient_id)

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
        """Consulta exames pendentes de um paciente."""
        calls["exams"].append(patient_id)

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
        """Recupera protocolos institucionais."""
        calls["protocols"].append(query)

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

    graph = build_workflow(
        get_patient_info,
        get_pending_exams,
        search_protocols,
    )

    return graph, calls


def test_patient_workflow(graph_with_tools):
    graph, calls = graph_with_tools

    result = graph.invoke({
        "question": "Quais exames estão pendentes para P001?"
    })

    context = result["result"]

    assert context["status"] == "context_ready"
    assert context["patient"]["patient_id"] == "P001"
    assert len(context["pending_exams"]) == 1
    assert context["sources"] == ["PRT-001"]

    assert calls["patient"] == ["P001"]
    assert calls["exams"] == ["P001"]
    assert len(calls["protocols"]) == 1


def test_patient_not_found(graph_with_tools):
    graph, calls = graph_with_tools

    result = graph.invoke({
        "question": "Quais exames estão pendentes para P999?"
    })

    assert result["result"]["status"] == "patient_not_found"

    # O fluxo deve parar antes dos exames e do RAG
    assert calls["patient"] == ["P999"]
    assert calls["exams"] == []
    assert calls["protocols"] == []


def test_general_question(graph_with_tools):
    graph, calls = graph_with_tools

    result = graph.invoke({
        "question": "Como validar uma recomendação clínica?"
    })

    context = result["result"]

    assert context["status"] == "context_ready"
    assert context["patient"] is None
    assert context["pending_exams"] == []
    assert context["sources"] == ["PRT-001"]

    # Não deve consultar o banco de pacientes
    assert calls["patient"] == []
    assert calls["exams"] == []
    assert len(calls["protocols"]) == 1


def test_empty_question(graph_with_tools):
    graph, calls = graph_with_tools

    result = graph.invoke({"question": "   "})

    assert result["result"]["status"] == "invalid_query"
    assert calls["patient"] == []
    assert calls["exams"] == []
    assert calls["protocols"] == []


def test_conflicting_patient_ids(graph_with_tools):
    graph, calls = graph_with_tools

    result = graph.invoke({
        "question": "Consultar os exames de P001",
        "patient_id": "P002",
    })

    assert result["result"]["status"] == "invalid_query"
    assert calls["patient"] == []
    assert calls["exams"] == []
    assert calls["protocols"] == []


def test_multiple_patient_ids(graph_with_tools):
    graph, calls = graph_with_tools

    result = graph.invoke({
        "question": "Comparar os pacientes P001 e P002"
    })

    assert result["result"]["status"] == "invalid_query"
    assert calls["patient"] == []
    assert calls["exams"] == []
    assert calls["protocols"] == []