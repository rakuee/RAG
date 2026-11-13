# rag_chain.py
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectorstore import load_vectorstore

load_dotenv()  # carga GOOGLE_API_KEY desde .env

NO_ANSWER_STRING = (
    "No dispongo de información suficiente en los documentos proporcionados "
    "para responder a esta pregunta."
)

SYSTEM_PROMPT = f"""Eres un asistente que responde ÚNICAMENTE basándose en el contexto recuperado de documentos internos.

Reglas estrictas:
1. Si el contexto contiene información que responde DIRECTAMENTE a la pregunta, úsala y cita la fuente.
2. Si el contexto es parcial o no responde a la pregunta, responde exactamente con esta frase: "{NO_ANSWER_STRING}"
3. NO inventes, NO supongas y NO uses conocimiento externo al contexto.
4. Si usas el contexto, termina la respuesta indicando entre corchetes el nombre del archivo fuente: [nombre_archivo].
"""


def _normalize(text: str) -> str:
    """Minúsculas, sin puntuación ni espacios repetidos (para comparar frases)."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def is_no_answer(answer: str) -> bool:
    """True si el modelo respondió con la frase de 'sin información',
    aunque la haya puesto entre comillas o cambiado la puntuación."""
    return _normalize(NO_ANSWER_STRING) in _normalize(answer)


def format_docs(docs):
    return "\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'desconocido')}]\n{d.page_content}"
        for d in docs
    )


def _build_llm():
    """Instancia el LLM generador de Gemini."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontró GOOGLE_API_KEY. Crea un archivo .env en la carpeta del "
            "proyecto con la línea GOOGLE_API_KEY=tu_clave (ver README, paso 7)."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",          # modelo Flash: buen equilibrio velocidad/calidad
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=1024,
        # convert_system_message_to_human=True,  # descomenta solo si tu modelo no acepta SystemMessage
    )
    # Alternativas de modelo (elige una):
    # model="gemini-2.5-flash"        # versión estable anterior
    # model="gemini-3.7-flash"        # familia más reciente
    # model="models/gemini-flash-latest"  # alias "latest" (útil para prototipos)


@lru_cache(maxsize=None)
def build_rag_chain(k: int = 4):
    """Crea el retriever y la cadena de generación UNA sola vez.

    El resultado queda en caché: el modelo de embeddings, ChromaDB y el LLM
    no se vuelven a cargar en cada pregunta.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Contexto:\n{context}\n\nPregunta: {question}"),
    ])

    generation_chain = prompt | _build_llm() | StrOutputParser()
    return generation_chain, retriever


def ask(question: str, k: int = 4):
    """Devuelve (respuesta, fragmentos recuperados). Los fragmentos se adjuntan para auditoría."""
    generation_chain, retriever = build_rag_chain(k)

    # Una sola búsqueda: los mismos fragmentos sirven de contexto y de fuentes.
    docs = retriever.invoke(question)
    answer = generation_chain.invoke(
        {"context": format_docs(docs), "question": question}
    )

    fuentes = [
        {
            "archivo": d.metadata.get("source", "desconocido"),
            "fragmento": d.page_content[:300],
        }
        for d in docs
    ]
    return answer, fuentes
