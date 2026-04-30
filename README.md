
# RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot for asking questions over uploaded PDF documents.
<!-- ...existing content... -->
## Demo

Here's a screen recording demo of the RAG Chatbot in action:
https://github.com/user-attachments/assets/e4f01801-6dad-47ae-a60c-b7ef70a2c98c


<!-- ...existing content... -->
## 📌 Project Overview

This project implements a complete RAG pipeline:

- Upload and process PDF documents  
- Retrieve relevant context using vector search  
- Generate answers using an LLM  
- Evaluate responses using a grounding score  

---

## 🧠 Key Features

### Document Understanding
- Upload multiple PDFs
- Automatic text extraction and chunking

### Semantic Retrieval
- Uses FAISS vector database
- Retrieves top-k relevant chunks per query

### AI-Powered Responses
- LLM generates answers using retrieved context
- Handles conversational queries with memory

### Evaluation System
- Computes Grounding Score (0–1)
- Measures alignment between response and context
- Labels responses: Strong / Moderate / Weak

### Logging
- Stores questions, answers, and scores
- Export results as CSV

---

## Folder Structure

- `app.py` - main Streamlit application file
- `requirements.txt` - Python dependencies
- `doc_files/` - directory used to store uploaded PDF files
- `ragchat/` - local virtual environment folder (not part of the app code)
- `.env` - environment file loaded by the app for secrets/configuration
---

## 🔄 How It Works

PDF → Chunk → Embedding → FAISS  
User Query → Retrieve → Context → LLM → Answer → Evaluation  

---

## 📊 Evaluation

Grounding Score interpretation:

- 0.55 – 1.0 → Strong  
- 0.30 – 0.55 → Moderate  
- 0.00 – 0.30 → Weak  

---
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
