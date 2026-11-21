import torch

from src.llm.model_provider import LocalQwenProvider


def main():
    provider = LocalQwenProvider()

    question = (
        "Um exame que não consta no prontuário "
        "pode ser considerado automaticamente pendente?"
    )

    context = (
        "PRT-001: Um exame sem registro não pode ser "
        "classificado automaticamente como pendente. "
        "É necessário consultar os registros de exames."
    )

    print("\nPERGUNTA:")
    print(question)

    response = provider.generate(
        question=question,
        context=context,
        max_new_tokens=120,
    )

    print("\nRESPOSTA:")
    print(response)

    if hasattr(torch.mps, "current_allocated_memory"):
        memory_gb = (
            torch.mps.current_allocated_memory() / 1024**3
        )
        print(f"\nMemória MPS alocada: {memory_gb:.2f} GB")


if __name__ == "__main__":
    main()