# diagnose.py
import os
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
)

DOCS_DIR = "docs"

for f in sorted(os.listdir(DOCS_DIR)):
    path = os.path.join(DOCS_DIR, f)
    if not os.path.isfile(path):
        print(f"[SKIP dir]  {f}")
        continue
    ext = os.path.splitext(f)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8-sig")
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(path)
        else:
            print(f"[SKIP ext]  {f}  (extensión no soportada: '{ext}')")
            continue
        docs = loader.load()
        chars = sum(len(d.page_content) for d in docs)
        flag = "  ⚠️ contenido casi vacío (¿PDF escaneado/cifrado?)" if chars < 50 else ""
        print(f"[OK]        {f}: {len(docs)} secciones/páginas, {chars} caracteres{flag}")
    except Exception as e:
        print(f"[ERROR]     {f}: {type(e).__name__}: {e}")