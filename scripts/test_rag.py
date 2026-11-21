from src.rag.vector_store import get_vector_store


def main():
    vector_store = get_vector_store()

    questions = [
        "Como identificar exames pendentes?",
        "Quais exames são importantes no acompanhamento do diabetes tipo 2?",
        "O assistente pode prescrever medicamentos sem aprovação médica?",
    ]

    for question in questions:
        print(f"\nPERGUNTA: {question}")
        print("-" * 60)

        results = vector_store.similarity_search_with_score(
            query=question,
            k=3,
        )

        for document, distance in results:
            print(
                f"Protocolo: "
                f"{document.metadata['protocol_id']}"
            )
            print(f"Distância: {distance:.4f}")
            print(f"Trecho: {document.page_content[:250]}")
            print("-" * 40)


if __name__ == "__main__":
    main()