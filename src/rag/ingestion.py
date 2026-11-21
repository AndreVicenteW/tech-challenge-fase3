from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.database.connection import PROJECT_ROOT

PROTOCOLS_DIR = PROJECT_ROOT / "data" / "protocols"


def load_protocols() -> list[Document]:
    documents = []

    for path in sorted(PROTOCOLS_DIR.glob("PRT-*.md")):
        content = path.read_text(encoding="utf-8")

        if not content.strip():
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "protocol_id": path.stem,
                    "source": str(path.relative_to(PROJECT_ROOT)),
                },
            )
        )

    return documents


def split_protocols(
    documents: list[Document],
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks