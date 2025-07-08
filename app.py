import streamlit as st
import requests
import os
from langchain.prompts import PromptTemplate

# ------------------------ CONFIG ------------------------
BACKEND_URL = os.getenv("HF_BACKEND_URL")  # Store this securely in Render

# Submit
if st.button("📖 Get Answer"):
    if not question.strip():
        st.warning("⚠️ Please enter a valid question.")
    elif BACKEND_URL is None:
        st.error("❌ Backend URL not found. Set the 'HF_BACKEND_URL' in Render.")
    else:
        with st.spinner("🕉️ Fetching wisdom from scriptures..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"question": question},  # only send user input
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
