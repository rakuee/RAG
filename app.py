# app.py
import streamlit as st
from rag_chain import ask, NO_ANSWER_STRING

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚")
st.title("📚 Asistente de Documentos (RAG)")
st.caption("Haz preguntas sobre los documentos indexados y consulta las fuentes exactas.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre los documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos..."):
            try:
                answer, fuentes = ask(prompt)
            except Exception as e:
                st.error(f"Error al consultar el modelo: {e}")
                st.stop()

        st.markdown(answer)

        if answer.strip() == NO_ANSWER_STRING:
            st.warning(
                "⚠️ El sistema no encontró evidencia suficiente en los documentos indexados."
            )
        else:
            with st.expander("📎 Ver fuentes utilizadas"):
                for i, f in enumerate(fuentes, 1):
                    st.markdown(f"**{i}. {f['archivo']}**")
                    st.caption(f"...{f['fragmento']}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})