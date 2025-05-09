# RAG Chat with BGE-M3

A Retrieval-Augmented Generation (RAG) system built with Streamlit, Ollama, and BGE-M3 model for embeddings. This application allows users to upload PDF documents and chat with an AI that can reference the content of those documents.

## Features

- Chat with your preferred Ollama models
- Upload PDF documents for context-aware answers using BGE-M3 embeddings
- Store and retrieve chat history
- Simple, streamlined interface

## Requirements

- [Ollama](https://ollama.ai/) installed locally
- BGE-M3 model pulled in Ollama (`ollama pull bge-m3:latest`)
- Python 3.8+

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure Ollama is running:
   ```
   ollama serve
   ```
4. Pull the necessary models:
   ```
   ollama pull bge-m3:latest
   ollama pull qwen2.5:latest  # or any other chat model you prefer
   ```

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```
2. Upload PDF documents using the sidebar
3. Toggle "PDF Chat" to enable RAG functionality
4. Start chatting

## Components

- `app.py`: Main Streamlit application file
- `database_operations.py`: SQLite database operations
- `pdf_handler.py`: PDF processing utilities
- `vectordb_handler.py`: Vector database for document embeddings
- `utils.py`: Utility functions

## How It Works

1. Upload PDFs: The system extracts text and breaks it into chunks
2. Embeddings: Text chunks are converted to vector embeddings using BGE-M3
3. Storage: Embeddings are stored in a ChromaDB vector database
4. Retrieval: When asking a question, the system retrieves relevant document chunks
5. Generation: Your selected Ollama model generates answers based on the retrieved chunks
6. Chat History: All conversations are stored in SQLite for future reference

## Troubleshooting

If you encounter a dimension mismatch error, you may need to:
1. Ensure your embedding model is consistent (BGE-M3 used for both embedding and retrieval)
2. Delete the existing ChromaDB database after closing Streamlit:
   ```
   # Delete the ChromaDB database to start fresh
   Remove-Item -Recurse -Force chroma_db  # Windows
   rm -rf chroma_db                      # Linux/Mac
   ```

## Folder Structure

- `chat_sessions/`: SQLite database storage
- `chroma_db/`: Vector database storage
- `pdfs/`: Directory for PDF storage
- `chat_icons/`: User and bot avatars

## Configuration

Edit `config.yaml` to modify:
- Embedding model
- Chat memory length
- Number of retrieved documents
- Text chunking parameters
