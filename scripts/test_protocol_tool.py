from src.tools.protocol_tools import create_protocol_tool


def main():
    search_tool = create_protocol_tool(k=4)

    questions = [
        "Como identificar exames pendentes?",
        "Quais exames são relevantes para acompanhar o diabetes tipo 2?",
        "O assistente pode prescrever medicamentos?",
    ]

    for question in questions:
        result = search_tool.invoke({"query": question})

        print(f"\nPERGUNTA: {question}")
        print(f"Status: {result['status']}")
        print(f"Chunks: {result.get('count', 0)}")

        for match in result["matches"]:
            print(f"\nProtocolo: {match['protocol_id']}")
            print(f"Distância: {match['distance']:.4f}")
            print(f"Fonte: {match['source']}")
            print(f"Conteúdo: {match['content'][:250]}")


if __name__ == "__main__":
    main()