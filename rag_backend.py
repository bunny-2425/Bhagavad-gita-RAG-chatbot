# rag_backend.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load tokenizer and model (Gemma 2B)
model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Inference function
def generate_response(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load ChromaDB
vectorstore = Chroma(persist_directory="./chromadb",
                     embedding_function=HuggingFaceEmbeddings(
                         model_name="sentence-transformers/all-MiniLM-L6-v2"
                     ))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

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

# LangChain Chain to bind together
llm_chain = prompt_extract | RunnableLambda(lambda x: generate_response(x)) | StrOutputParser()

def run_rag_pipeline(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    result = llm_chain.invoke({"context": context, "question": query})
    return result

# Optional test entry point
if __name__ == "__main__":
    question = "What is the essence of Bhagavad Gita?"
    print(run_rag_pipeline(question))