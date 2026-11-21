from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graph.workflow import build_workflow
from src.llm.ollama_provider import OllamaProvider
from src.tools.patient_tools import create_patient_tools
from src.tools.protocol_tools import create_protocol_tool
from src.audit.store import AuditStore


def main():
    patient_tool, exams_tool = create_patient_tools()
    audit_store = AuditStore()

    graph = build_workflow(
        patient_tool=patient_tool,
        exams_tool=exams_tool,
        protocol_tool=create_protocol_tool(k=4),
        llm_provider=OllamaProvider(),
        checkpointer=InMemorySaver(),
        audit_store=audit_store,
    )

    config = {
        "configurable": {
            "thread_id": str(uuid4()),
        }
    }

    question = "Quais exames estão pendentes para P001?"

    request_id = str(uuid4())

    state = graph.invoke(
        {
            "question": question,
            "request_id": request_id,
        },
        config=config,
    )

    if "__interrupt__" not in state:
        print("Fluxo encerrado sem revisão:")
        print(state["result"])
        return

    review = state["__interrupt__"][0].value

    print("\n=== RASCUNHO ===")
    print(review["draft"])

    print("\n=== EXAMES CONFIRMADOS ===")
    print(review["verified_exams"])

    print("\n=== FONTES ===")
    print(review["sources"])

    choice = input(
        "\nAprovar o rascunho? (s/N): "
    ).strip().lower()

    final_state = graph.invoke(
        Command(resume={"approved": choice == "s"}),
        config=config,
    )

    print("\n=== DEBUG AUDITORIA ===")
    print("Request ID:", request_id)
    print("Request ID no estado:", final_state.get("request_id"))
    print("Auditoria registrada:", final_state.get("audit_recorded"))
    print("Resultado final:", final_state.get("result"))
    print("Banco de auditoria:", audit_store.db_path)

    audit_event = audit_store.get(request_id)

    if audit_event is None:
        print("\nNenhum registro encontrado para esse request_id.")
    else:
        print("\n=== REGISTRO DE AUDITORIA ===")
        print(f"ID: {audit_event['request_id']}")
        print(f"Status: {audit_event['status']}")
        print(f"Fontes: {audit_event['sources']}")
        print(f"Validação: {audit_event['validation']}")

    print("\n=== RESULTADO FINAL ===")
    print(final_state["result"])


if __name__ == "__main__":
    main()