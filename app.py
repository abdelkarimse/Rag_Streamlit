# Local Multimodal AI Chat - Multimodal chat application with local models
# Copyright (C) 2024 Leon Sander
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import streamlit as st
import os
import uuid
import sqlite3
from vectordb_handler import load_vectordb
from pdf_handler import add_documents_to_db
from database_operations import init_db, save_text_message, load_messages, get_all_chat_history_ids, delete_chat_history, load_last_k_text_messages_ollama
from utils import get_avatar, list_ollama_models, load_config
import requests

# Define CSS styling directly
css = """
<style>
.user-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
}
.bot-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
}
</style>
"""

# Configuration
config = load_config()

# Create necessary directories
os.makedirs("chat_sessions", exist_ok=True)
os.makedirs("pdfs", exist_ok=True)
os.makedirs("chat_icons", exist_ok=True)

# Initialize database
init_db()

def get_session_key():
    """Get or create a session key for the current chat session."""
    if st.session_state.session_key == "new_session":
        if st.session_state.new_session_key is None:
            st.session_state.new_session_key = str(uuid.uuid4())
        return st.session_state.new_session_key
    return st.session_state.session_key

def get_user_id():
    """Get the current user ID (default is 1 for simplicity)."""
    return 1

def delete_chat_session_history():
    """Delete the current chat session history."""
    delete_chat_history(get_session_key(), get_user_id())
    st.session_state.session_key = "new_session"
    st.session_state.new_session_key = None

def clear_cache():
    """Clear Streamlit cache."""
    st.cache_resource.clear()

def toggle_pdf_chat():
    """Toggle PDF chat mode."""
    st.session_state.pdf_chat = True
    clear_cache()

def detoggle_pdf_chat():
    """Disable PDF chat mode."""
    st.session_state.pdf_chat = False

def list_model_options():
    """List available Ollama models."""
    ollama_options = list_ollama_models()
    if not ollama_options:
        st.warning("No Ollama models available. Please choose one from https://ollama.com/library and pull it with 'ollama pull <model_name>'")
    return ollama_options

def update_model_options():
    """Update the available model options."""
    st.session_state.model_options = list_model_options()

def chat_with_rag(user_input, chat_history):
    """Chat with RAG system when PDF mode is enabled."""
    vector_db = load_vectordb()
    retrieved_documents = vector_db.similarity_search(
        user_input, 
        k=config["chat_config"]["number_of_retrieved_documents"]
    )
    context = "\n".join([item.page_content for item in retrieved_documents])
    
    messages = [{"role": "system", "content": f"You are a helpful assistant. Base your answers on the following context: {context}"}]
    
    # Add chat history
    for msg in chat_history:
        role = "user" if msg["sender_type"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})
    
    # Make API call to Ollama
    data = {
        "model": st.session_state.model_to_use,
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post(
            url="http://127.0.0.1:11434/api/chat", 
            json=data
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def chat_without_rag(user_input, chat_history):
    """Regular chat without RAG."""
    messages = []
    
    # Add chat history
    for msg in chat_history:
        role = "user" if msg["sender_type"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})
    
    # Make API call to Ollama
    data = {
        "model": st.session_state.model_to_use,
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post(
            url="http://127.0.0.1:11434/api/chat", 
            json=data
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("RAG Chat with BGE-M3")
    st.write(css, unsafe_allow_html=True)
    
    # Initialize session state
    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect("chat_sessions/chat_history.db", check_same_thread=False)
        st.session_state.pdf_uploader_key = 0
        st.session_state.pdf_chat = False
        st.session_state.model_options = list_model_options()
        st.session_state.model_to_use = st.session_state.model_options[0] if st.session_state.model_options else "qwen2.5:latest"
    
    # Update session tracker
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    # Sidebar configuration
    st.sidebar.title("Chat Sessions")
    
    # Get all chat sessions
    chat_sessions = ["new_session"] + get_all_chat_history_ids(get_user_id())
    
    # Select session
    try:
        index = chat_sessions.index(st.session_state.session_index_tracker)
    except ValueError:
        st.session_state.session_index_tracker = "new_session"
        index = chat_sessions.index(st.session_state.session_index_tracker)
        clear_cache()

    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    
    # Model selection
    st.sidebar.selectbox(label="Select a Model", options=st.session_state.model_options, key="model_to_use")
    
    # Toggle PDF chat
    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False, on_change=clear_cache)
    
    # Delete chat session
    st.sidebar.button("Delete Chat Session", on_click=delete_chat_session_history)
    
    # PDF file uploader
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload PDFs for RAG", 
        accept_multiple_files=True, 
        key=st.session_state.pdf_uploader_key, 
        type=["pdf"], 
        on_change=toggle_pdf_chat
    )

    # Process uploaded PDFs
    if uploaded_pdf:
        with st.spinner("Processing PDFs..."):
            add_documents_to_db(uploaded_pdf)
            st.session_state.pdf_uploader_key += 2
    
    # Chat container
    chat_container = st.container()
    
    # User input
    user_input = st.chat_input("Type your message here", key="user_input")
    
    # Process user input
    if user_input:
        if st.session_state.pdf_chat:
            llm_answer = chat_with_rag(
                user_input, 
                load_last_k_text_messages_ollama(get_session_key(), config["chat_config"]["chat_memory_length"])
            )
        else:
            llm_answer = chat_without_rag(
                user_input, 
                load_last_k_text_messages_ollama(get_session_key(), config["chat_config"]["chat_memory_length"])
            )
        
        # Save messages to database
        save_text_message(get_session_key(), "user", user_input, get_user_id())
        save_text_message(get_session_key(), "assistant", llm_answer, get_user_id())
    
    # Display chat history
    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key is not None):
        with chat_container:
            chat_history_messages = load_messages(get_session_key(), get_user_id())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
        
        if (st.session_state.session_key == "new_session") and (st.session_state.new_session_key is not None):
            st.rerun()

if __name__ == "__main__":
    main()
