import streamlit as st
import requests
from langchain.prompts import PromptTemplate

# ------------------------
# CONFIG
# ------------------------
BACKEND_URL = "https://<your-hf-username>.hf.space/generate"  # ⛳ Replace with your actual backend URL

# ------------------------
# PromptTemplate Setup
# ------------------------
prompt_template = PromptTemplate.from_template(
    """You are an AI assistant that provides accurate and context-aware responses based on the given retrieved documents.
Use the context below to answer the question concisely and accurately in any language and while mentioning the book
from which the answer is retrieved. Mention in detail if it is from Bhagavad Gita or any volume of Srimad Bhagavatam.

Context:
{context}

Question:
{question}

Greeting from Shri Krishna for every user's first query and give the shloka or verse example from which the answer is created:
Answer:"""
)

# ------------------------
# Streamlit Page Settings
# ------------------------
st.set_page_config(page_title="Shri Krishna RAG Chatbot", layout="centered")
st.title("🕉️ श्रीकृष्ण संवाद: Bhagavad Gita & Srimad Bhagavatam Chatbot")

# ------------------------
# Input Field
# ------------------------
question = st.text_input("🙏 Ask your spiritual question:")

# ------------------------
# Ask Button
# ------------------------
if st.button("📖 Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("🕉️ Fetching wisdom from scriptures..."):
            try:
                formatted_prompt = prompt_template.format(context="", question=question)

                response = requests.post(BACKEND_URL, json={"question": formatted_prompt}, timeout=20)

                if response.status_code == 200:
                    reply = response.json().get("response", "❌ No response received.")
                else:
                    reply = f"⚠️ Error {response.status_code} from backend."

            except requests.exceptions.Timeout:
                reply = "⏳ Request timed out. Please try again later."
            except requests.exceptions.ConnectionError:
                reply = "❌ Cannot connect to backend. Is the Hugging Face Space running?"
            except Exception as e:
                reply = f"⚠️ Unexpected error: {str(e)}"

        st.markdown(f"**Shri Krishna Bot:** {reply}")
