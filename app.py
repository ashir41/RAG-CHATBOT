import os
import shutil
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Evaluation imports
from sentence_transformers import SentenceTransformer, util

# ------------------- UI CONFIG -------------------
st.set_page_config(page_title="RAG Chatbot (Improved Evaluation)", layout="wide")
st.title("📄 Chat with your Documents + Grounding Score")

# ------------------- RESET -------------------
if st.button("🔄 Reset System"):
    st.session_state.clear()
    st.rerun()

# ------------------- SESSION -------------------
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "evaluation_logs" not in st.session_state:
    st.session_state.evaluation_logs = []

if "eval_model" not in st.session_state:
    st.session_state.eval_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- EVALUATION -------------------
def evaluate_response(answer, query, retrieved_docs):
    model = st.session_state.eval_model

    if not retrieved_docs:
        return 0.0

    # 🔥 Combine query + answer for better semantic match
    combined_text = query + " " + answer
    emb_answer = model.encode(combined_text, convert_to_tensor=True)

    scores = []

    for doc in retrieved_docs:
        # 🔥 Limit chunk size to avoid noise
        chunk = doc.page_content[:400]

        emb_doc = model.encode(chunk, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_answer, emb_doc))
        scores.append(sim)

    return max(scores)

# ------------------- PROCESS DOCUMENT -------------------
def process_document(path):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # 🔥 Smaller chunks (IMPORTANT FIX)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(docs, embeddings)

    # ------------------- TOOL -------------------
    @tool
    def retrieve_context(query: str):
        """Retrieve relevant document chunks from vector database"""
        results = vector_db.similarity_search(query, k=4)

        st.session_state.last_retrieved_docs = results

        context = "\n\n".join([doc.page_content for doc in results])
        return context

    # ------------------- LLM -------------------
    llm = ChatGroq(model="openai/gpt-oss-20b")

    system_prompt = """You are a helpful AI assistant.

Use the provided context to answer.
If the answer is not found, say "I don't know".
Be concise and accurate.
"""

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

# ------------------- FILE UPLOAD -------------------
if not st.session_state.document_uploaded:
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            path = "./doc_files/"

            if os.path.exists(path):
                shutil.rmtree(path)

            os.makedirs(path, exist_ok=True)

            st.session_state.messages = []
            st.session_state.evaluation_logs = []

            for file in uploaded_files:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.success("Documents processed!")
            st.rerun()

# ------------------- CHAT -------------------
if st.session_state.document_uploaded and st.session_state.agent:

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.evaluation_logs = []
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask something about your documents...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke(
                {"messages": st.session_state.messages},
                {"configurable": {"thread_id": "chat_session"}}
            )

        answer = response["messages"][-1].content

        # ------------------- EVALUATION -------------------
        retrieved_docs = st.session_state.get("last_retrieved_docs", [])
        score = evaluate_response(answer, query, retrieved_docs)

        # 🔥 Better interpretation scale
        if score > 0.55:
            label = "Strong"
        elif score > 0.30:
            label = "Moderate"
        else:
            label = "Weak"

        st.session_state.evaluation_logs.append({
            "question": query,
            "answer": answer,
            "grounding_score": score,
            "grounding_label": label
        })

        # ------------------- DISPLAY -------------------
        with st.chat_message("assistant"):
            st.markdown(answer)

            with st.expander("📊 Evaluation Info"):
                st.write(f"Grounding Score: {score:.2f}")
                st.write(f"Grounding Level: {label}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

# ------------------- DOWNLOAD -------------------
if st.session_state.evaluation_logs:
    df = pd.DataFrame(st.session_state.evaluation_logs)

    st.download_button(
        "📥 Download Evaluation Results",
        df.to_csv(index=False),
        "evaluation_results.csv",
        "text/csv"
    )