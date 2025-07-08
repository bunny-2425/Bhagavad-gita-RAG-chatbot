import streamlit as st
import requests
import os
from langchain.prompts import PromptTemplate

# ------------------------ CONFIG ------------------------
BACKEND_URL = os.getenv("HF_BACKEND_URL")  # Store this securely in Render

# ------------------------ Prompt Template ------------------------
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

# ------------------------ Streamlit UI ------------------------
st.set_page_config(page_title="Shri Krishna RAG Chatbot", layout="centered")
st.title("🕉️ श्रीकृष्ण संवाद: Bhagavad Gita & Srimad Bhagavatam Chatbot")

# Input field
question = st.text_input("🙏 Ask your spiritual question:")

# Submit
if st.button("📖 Get Answer"):
    if not question.strip():
        st.warning("⚠️ Please enter a valid question.")
    elif BACKEND_URL is None:
        st.error("❌ Backend URL not found. Set the 'HF_BACKEND_URL' in Render.")
    else:
        with st.spinner("🕉️ Fetching wisdom from scriptures..."):
            try:
                # Format the prompt using LangChain template
                formatted_prompt = prompt_template.format(context="", question=question)

                # Send POST request to FastAPI backend
                response = requests.post(
                    BACKEND_URL,
                    json={"question": formatted_prompt},
                    timeout=30,
                )

                if response.status_code == 200:
                    answer = response.json().get("response", "❌ No response found.")
                else:
                    answer = f"⚠️ Backend error: {response.status_code}"

            except requests.exceptions.Timeout:
                answer = "⏳ Request timed out. Please try again later."
            except requests.exceptions.ConnectionError:
                answer = "❌ Could not connect to the backend. Is Hugging Face Space running?"
            except Exception as e:
                answer = f"⚠️ Unexpected error: {str(e)}"

        st.markdown(f"**🕉️ Shri Krishna Bot:**\n\n{answer}")
