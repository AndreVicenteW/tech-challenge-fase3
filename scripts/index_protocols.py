from src.rag.ingestion import (
    load_protocols,
    split_protocols,
)
from src.rag.vector_store import get_vector_store


def main():
    documents = load_protocols()

    if not documents:
        raise RuntimeError(
            "Nenhum protocolo encontrado em data/protocols."
        )

    chunks = split_protocols(documents)

    vector_store = get_vector_store()

    # Recriar somente os registros da coleção de protocolos
    existing_ids = vector_store.get(include=[])["ids"]

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    # IDs reproduzíveis dos chunks
    ids = [
        f"{chunk.metadata['protocol_id']}-{index:04d}"
        for index, chunk in enumerate(chunks)
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(f"Protocolos carregados: {len(documents)}")
    print(f"Chunks indexados: {len(chunks)}")
    print("Indexação concluída!")


if __name__ == "__main__":
    main()