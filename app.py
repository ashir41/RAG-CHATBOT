import os
import streamlit as st
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

# ------------------- UI CONFIG -------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 Chat with your Documents")

# ------------------- SESSION STATE -------------------
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------- DOCUMENT PROCESSING -------------------
def process_document(path):
    # Load PDFs
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector store (FAISS)
    vector_db = FAISS.from_documents(docs, embeddings)

    # ------------------- TOOL -------------------
    @tool
    def retrieve_context(query: str):
        """Retrieve relevant document chunks"""
        results = vector_db.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in results])
        return context

    # ------------------- LLM -------------------
    llm = ChatGroq(model="llama3-8b-8192")

    # ------------------- PROMPT -------------------
    system_prompt = """You are a helpful AI assistant.

Use the retrieved context to answer questions.
If the answer is not in the context, say you don't know.

Also consider previous conversation history when answering.
"""

    # ------------------- MEMORY -------------------
    memory = InMemorySaver()

    # ------------------- AGENT -------------------
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
            os.makedirs(path, exist_ok=True)

            for file in uploaded_files:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.success("Documents processed!")
            st.rerun()

# ------------------- CHAT UI -------------------
if st.session_state.document_uploaded and st.session_state.agent:

    # Clear chat button
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    query = st.chat_input("Ask something about your documents...")

    if query:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # Get response
        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke(
                {"messages": st.session_state.messages},
                {"configurable": {"thread_id": "chat_session"}}
            )

        answer = response["messages"][-1].content

        # Show AI response
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )