import json

from src.graph.workflow import build_workflow
from src.tools.patient_tools import create_patient_tools
from src.tools.protocol_tools import create_protocol_tool
from src.llm.ollama_provider import OllamaProvider


def main():
    patient_tool, exams_tool = create_patient_tools()

    graph = build_workflow(
        patient_tool,
        exams_tool,
        create_protocol_tool(k=4),
    )

    question = "Quais exames estão pendentes para P001?"

    state = graph.invoke({"question": question})
    result = state["result"]

    if result["status"] != "context_ready":
        print(result)
        return

    # Dados verificados pelas ferramentas
    verified = {
        "patient": result["patient"],
        "pending_exams": result["pending_exams"],
        "protocols": [
            {
                "source": item["source"],
                "content": item["content"],
            }
            for item in result["protocols"]
        ],
    }

    context = json.dumps(
        verified,
        ensure_ascii=False,
        indent=2,
    )

    print("=== EXAMES CONFIRMADOS NO SQLITE ===")
    print(verified["pending_exams"])

    provider = OllamaProvider()

    draft = provider.generate_draft(
        question=question,
        context=context,
    )

    print("\n=== RASCUNHO DA LLM — NÃO VALIDADO ===")
    print(draft)


if __name__ == "__main__":
    main()