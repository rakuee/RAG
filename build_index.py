# build_index.py
"""Crea (o vuelve a crear) el índice vectorial de ChromaDB a partir de los documentos de docs/.

Uso:
    python build_index.py                 # reconstruye el índice desde cero
    python build_index.py --query "texto"  # reconstruye y hace una búsqueda de prueba
"""
import argparse
from collections import Counter

import chromadb

from vectorstore import PERSIST_DIR, COLLECTION_NAME, build_vectorstore


def reset_collection():
    """Elimina la colección anterior para que los fragmentos no se dupliquen."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    existentes = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if COLLECTION_NAME in existentes:
        client.delete_collection(COLLECTION_NAME)
        print(f"Colección anterior '{COLLECTION_NAME}' eliminada.")
    else:
        print(f"No había una colección '{COLLECTION_NAME}'; se creará una nueva.")


def main():
    parser = argparse.ArgumentParser(description="Construye el índice vectorial del RAG.")
    parser.add_argument("--query", help="Pregunta de prueba para verificar la búsqueda semántica.")
    parser.add_argument("--k", type=int, default=4, help="Fragmentos a mostrar en la prueba (por defecto 4).")
    args = parser.parse_args()

    print("1/3 Limpiando índice anterior...")
    reset_collection()

    print("2/3 Cargando documentos, generando embeddings e indexando...")
    vectorstore = build_vectorstore()

    print("3/3 Verificando...")
    datos = vectorstore.get(include=["metadatas"])
    por_archivo = Counter(m.get("source", "desconocido") for m in datos["metadatas"])
    print(f"\nTotal de fragmentos en el índice: {len(datos['ids'])}")
    for archivo, n in sorted(por_archivo.items()):
        print(f"  - {archivo}: {n} fragmentos")

    if args.query:
        print(f"\nBúsqueda de prueba: \"{args.query}\"")
        resultados = vectorstore.similarity_search_with_relevance_scores(args.query, k=args.k)
        for i, (doc, score) in enumerate(resultados, 1):
            texto = doc.page_content.replace("\n", " ")[:150]
            print(f"  {i}. [{doc.metadata.get('source')}] relevancia={score:.3f}  {texto}...")

    print("\nÍndice listo. Ejecuta la app con:  streamlit run app.py")


if __name__ == "__main__":
    main()
