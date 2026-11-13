# rag_chain.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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


def format_docs(docs):
    return "\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'desconocido')}]\n{d.page_content}"
        for d in docs
    )


def _build_llm():
    """Instancia el LLM generador de Gemini."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",          # modelo Flash: buen equilibrio velocidad/calidad
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0,
        max_output_tokens=1024,
        # convert_system_message_to_human=True,  # descomenta solo si tu modelo no acepta SystemMessage
    )
    # Alternativas de modelo (elige una):
    # model="gemini-2.5-flash"        # versión estable anterior
    # model="gemini-3.7-flash"        # familia más reciente
    # model="models/gemini-flash-latest"  # alias "latest" (útil para prototipos)


def build_rag_chain(k: int = 4):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Contexto:\n{context}\n\nPregunta: {question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(question: str):
    """Devuelve (respuesta, fuentes). Las fuentes siempre se adjuntan para auditoría."""
    chain, retriever = build_rag_chain()
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    fuentes = [
        {
            "archivo": d.metadata.get("source", "desconocido"),
            "fragmento": d.page_content[:300],
        }
        for d in docs
    ]
    return answer, fuentes