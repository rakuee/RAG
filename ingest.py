# ingest.py
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "docs"


def load_documents():
    """Carga todos los PDF, TXT y MD de la carpeta docs/."""
    documentos = []
    if not os.path.isdir(DOCS_DIR):
        raise FileNotFoundError(f"No existe la carpeta '{DOCS_DIR}'. Crea una y añade tus documentos.")

    for filename in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif filename.lower().endswith(".md"):
            loader = UnstructuredMarkdownLoader(path)
        else:
            continue
        documentos.extend(loader.load())

    if not documentos:
        raise ValueError(f"No se encontraron documentos válidos en '{DOCS_DIR}'.")
    return documentos


def split_documents(documentos):
    """Divide los documentos en fragmentos con solapamiento."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documentos)
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(
            chunk.metadata.get("source", "desconocido")
        )
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Documentos cargados: {len(docs)} | Chunks generados: {len(chunks)}")