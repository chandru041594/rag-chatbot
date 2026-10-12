import streamlit as st
from rag_pipeline import rag_answer

st.title("📄 RAG Chatbot (PDF + Free Models)")

query = st.text_input("Ask a question about your PDF:")
if query:
    answer = rag_answer(query)
    st.write("**Answer:**", answer)
