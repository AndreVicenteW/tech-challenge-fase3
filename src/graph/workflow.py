from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END

from src.graph.state import ClinicalState
from src.graph.nodes import ClinicalNodes
from src.graph.response_nodes import ResponseNodes
from src.audit.nodes import AuditNodes

def build_workflow(
    patient_tool: BaseTool,
    exams_tool: BaseTool,
    protocol_tool: BaseTool,
    llm_provider=None,
    checkpointer=None,
    audit_store=None,
):

    if llm_provider is not None and checkpointer is None:
        raise ValueError(
            "A revisão humana exige um checkpointer."
        )

    nodes = ClinicalNodes(
        patient_tool,
        exams_tool,
        protocol_tool,
    )

    builder = StateGraph(ClinicalState)

    builder.add_node("validate", nodes.validate)
    builder.add_node("patient", nodes.fetch_patient)
    builder.add_node("exams", nodes.fetch_exams)
    builder.add_node("protocols", nodes.fetch_protocols)
    builder.add_node("context", nodes.prepare_context)
    builder.add_node("failure", nodes.failure)

    if audit_store is not None:
        audit_nodes = AuditNodes(audit_store)

        builder.add_node(
            "audit",
            audit_nodes.record,
        )

    terminal = "audit" if audit_store is not None else END

    builder.add_edge(START, "validate")

    def route_request(state: ClinicalState) -> str:
        if state["status"] != "ready":
            return "failure"

        if state.get("patient_id"):
            return "patient"

        return "protocols"

    builder.add_conditional_edges(
        "validate",
        route_request,
        ["patient", "protocols", "failure"],
    )

    def route_patient(state: ClinicalState) -> str:
        if state["status"] != "ready":
            return "failure"

        return "exams"

    builder.add_conditional_edges(
        "patient",
        route_patient,
        ["exams", "failure"],
    )

    builder.add_conditional_edges(
        "exams",
        lambda state: (
            "protocols"
            if state["status"] == "ready"
            else "failure"
        ),
        ["protocols", "failure"],
    )

    builder.add_conditional_edges(
        "protocols",
        lambda state: (
            "context"
            if state["status"] == "ready"
            else "failure"
        ),
        ["context", "failure"],
    )

    if llm_provider is not None:
        response_nodes = ResponseNodes(llm_provider)

        builder.add_node("generate", response_nodes.generate)
        builder.add_node("human_review", response_nodes.human_review)

        builder.add_node(
            "validate_response",
            response_nodes.validate_response,
        )

        builder.add_edge("context", "generate")

        builder.add_conditional_edges(
            "generate",
            lambda state: (
                "validate_response"
                if state["status"] == "draft_ready"
                else "failure"
            ),
            ["validate_response", "failure"],
        )

        builder.add_conditional_edges(
            "validate_response",
            lambda state: (
                "human_review"
                if state["status"] == "validation_ready"
                else "failure"
            ),
            ["human_review", "failure"],
        )

        builder.add_edge("human_review", terminal)

    else:
        # Mantém o comportamento dos testes anteriores
        builder.add_edge("context", terminal)

    builder.add_edge("failure", terminal)

    if audit_store is not None:
        builder.add_edge("audit", END)

    return builder.compile(checkpointer=checkpointer)