# vectorstore.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_documents, split_documents

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore():
    """Indexa los documentos en ChromaDB (sobrescribe si ya existe)."""
    embeddings = _get_embeddings()
    docs = load_documents()
    chunks = split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Indexados {len(chunks)} chunks en ChromaDB ({PERSIST_DIR}).")
    return vectorstore


def load_vectorstore():
    """Carga el índice existente sin reconstruirlo."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )