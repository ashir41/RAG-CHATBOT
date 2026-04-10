# RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot for asking questions over uploaded PDF documents.

## Project Overview

This project lets you upload PDFs, index their text in a FAISS vector store, and query them with a conversational AI agent.
<!-- ...existing content... -->

## Demo

Here's a screen recording demo of the RAG Chatbot in action:

<video width="640" height="360" controls>
  <source src="./recording/streamlit-app-2026-04-10-18-34-21.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

<!-- ...existing content... -->
The app is built with:
- `streamlit` for the web interface
- `langchain` and `langchain-community` for document loading, splitting, embeddings, and retrieval
- `FAISS` for vector search
- `ChatGroq` for the LLM/chat agent
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings

## Folder Structure

- `app.py` - main Streamlit application file
- `requirements.txt` - Python dependencies
- `doc_files/` - directory used to store uploaded PDF files
- `ragchat/` - local virtual environment folder (not part of the app code)
- `.env` - environment file loaded by the app for secrets/configuration

## How the App Works

1. Upload PDF files through the Streamlit UI.
2. PDFs are saved into `doc_files/`.
3. The app loads all PDFs from `doc_files/` using `PyPDFDirectoryLoader`.
4. Documents are chunked with `RecursiveCharacterTextSplitter`.
5. Embeddings are generated with `HuggingFaceEmbeddings`.
6. The chunks are indexed with `FAISS`.
7. A LangChain agent uses `ChatGroq` to answer user questions with retrieved context.

## Setup Instructions

1. Activate your Python environment.

   If your environment is `ragchat`, run in bash:
   ```bash
   source /c/Users/User/Desktop/rag_Chatbot/ragchat/Scripts/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file if needed.
   - The app already loads environment variables via `python-dotenv`.
   - Add any provider-specific keys required by `ChatGroq` or other services.

## Running the App

From the project root:

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Usage

- Upload one or more PDF files using the file uploader.
- Wait for the documents to process.
- Ask questions in the chat interface.
- Use the clear chat button to reset the conversation.

## Notes

- Uploaded PDFs are stored in `doc_files/`.
- The app currently uses the model `llama-3.3-70b-versatile` via `ChatGroq`.
- If you want to reuse the same uploaded documents later, keep the `doc_files/` content intact.

## Troubleshooting

- If Streamlit fails to start, confirm the correct Python environment is active.
- If PDF processing fails, verify the uploaded files are valid PDFs.
- If the LLM provider requires credentials, add them to `.env`.

## License

This repository has no license specified. Add a license file if you want to share or publish the project.
