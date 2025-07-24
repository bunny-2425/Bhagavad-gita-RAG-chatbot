import streamlit as st
import requests
import os

# ------------------------ CONFIG ------------------------
BACKEND_URL = os.getenv("HF_BACKEND_URL")  # Must be set in Render

st.set_page_config(page_title="Shri Krishna RAG Chatbot", page_icon="🕉️")

st.title("🕉️ Shri Krishna RAG Chatbot")
st.markdown("Ask your question about **Bhagavad Gita** or **Sanatan Dharma**")

# Input
question = st.text_input("🙏 Ask your question:")

# Initialize session state
if "answer" not in st.session_state:
    st.session_state.answer = ""

# On Enter Key press or Button
if question:
    st.session_state.answer = ""  # Clear previous
    if BACKEND_URL is None:
        st.error("❌ Backend URL not found. Set the 'HF_BACKEND_URL' in Render.")
    else:
        with st.spinner("🕉️ Fetching wisdom from scriptures..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"question": question},
                    timeout=30,
                )
                if response.status_code == 200:
                    st.session_state.answer = response.json().get("response", "❌ No response.")
                else:
                    st.session_state.answer = f"⚠️ Backend error: {response.status_code}"
            except requests.exceptions.Timeout:
                st.session_state.answer = "⏳ Timeout. Try again."
            except requests.exceptions.ConnectionError:
                st.session_state.answer = "❌ Connection error. Is backend live?"
            except Exception as e:
                st.session_state.answer = f"⚠️ Unexpected error: {str(e)}"

# Display answer
if st.session_state.answer:
    st.markdown(f"**🕉️ Shri Krishna Bot:**\n\n{st.session_state.answer}")
