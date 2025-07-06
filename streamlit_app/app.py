import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load tokenizer and model
model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Create HF pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

# Load ChromaDB retriever
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = Chroma(persist_directory="chromadb", embedding_function=embedding_model).as_retriever()

# Prompt Template
prompt_extract = PromptTemplate.from_template(
    """You are an AI assistant that provides accurate and context-aware responses based on the given retrieved documents.
    Use the context below to answer the question concisely and accurately in any language and while mentioning the book
    from which the answer is retrieved mention in detail is it from bhagwad geeta or any volume of srimad bhagwatam.

    Context:
    {context}

    Question:
    {question}

    Greeting from Shri Krishna for every users first query and give the shloka or verse example from which the answer is created:
    Answer:"""
)

# LangChain chain
chain = prompt_extract | llm_pipeline

# Streamlit UI
st.set_page_config(page_title="Bhagavad Gita RAG Chatbot", layout="wide")
st.title("🕉️ Bhagavad Gita & Srimad Bhagavatam RAG Chatbot")
st.write("Ask any spiritual or scriptural question. Answers will be generated using retrieved documents and Gemma-2B.")

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_query = st.chat_input("Ask your question to Shri Krishna's AI...")

if user_query:
    # Retrieve documents
    docs = retriever.get_relevant_documents(user_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Invoke chain
    response = chain.invoke({"context": context, "question": user_query})

    # Save history
    st.session_state.history.append((user_query, response[0]['generated_text']))

# Display chat history
for q, a in st.session_state.history:
    st.chat_message("user").markdown(q)
    st.chat_message("assistant").markdown(a)