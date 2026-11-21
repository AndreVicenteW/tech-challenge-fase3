from src.graph.workflow import build_workflow
from src.tools.patient_tools import create_patient_tools
from src.tools.protocol_tools import create_protocol_tool


def main():
    patient_tool, exams_tool = create_patient_tools()
    protocol_tool = create_protocol_tool(k=4)

    graph = build_workflow(
        patient_tool,
        exams_tool,
        protocol_tool,
    )

    questions = [
        "Quais exames estão pendentes para o paciente P001?",
        "Quais exames estão pendentes para o paciente P999?",
        "Como funciona a validação de recomendações clínicas?",
    ]

    for question in questions:
        print(f"\nPERGUNTA: {question}")
        print("-" * 60)

        state = graph.invoke({"question": question})
        result = state["result"]

        print(f"Status: {result['status']}")

        if result["status"] == "context_ready":
            print(f"Paciente: {result['patient']}")
            print(f"Pendências: {result['pending_exams']}")
            print(f"Fontes: {result['sources']}")
            print(
                f"Chunks: {len(result['protocols'])}"
            )
        else:
            print(f"Mensagem: {result['message']}")


if __name__ == "__main__":
    main()