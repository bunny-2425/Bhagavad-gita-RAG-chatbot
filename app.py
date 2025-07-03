import streamlit as st

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("💬 RAG-based Chatbot")
user_input = st.text_input("Ask a question:")

if user_input:
    st.write(f"🔍 Searching knowledge base for: `{user_input}`")
    st.success("🚧 This is a placeholder. RAG response will appear here.")