from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils import load_config
import chromadb
import os

# Load configuration
config = load_config()

def get_ollama_embeddings():
    """
    Creates an instance of OllamaEmbeddings using the qwen model.
    
    Returns:
        OllamaEmbeddings: Embeddings model
    """
    return OllamaEmbeddings(model="bge-m3:latest")

def delete_existing_collection():
    """
    Deletes the existing ChromaDB collection.
    """
    persistent_client = chromadb.PersistentClient(path=config["chromadb"]["chromadb_path"])
    try:
        persistent_client.delete_collection(config["chromadb"]["collection_name"])
        print(f"Deleted existing collection: {config['chromadb']['collection_name']}")
    except Exception as e:
        print(f"No existing collection to delete: {str(e)}")

def load_vectordb():
    """
    Loads the vector database with the Ollama embeddings.
    
    Returns:
        Chroma: Vector database instance
    """
    # Create embeddings
    embeddings = get_ollama_embeddings()
    
    # Ensure the ChromaDB directory exists
    os.makedirs(config["chromadb"]["chromadb_path"], exist_ok=True)
    
    # Initialize the persistent client for ChromaDB
    persistent_client = chromadb.PersistentClient(path=config["chromadb"]["chromadb_path"])

    # Delete existing collection if it exists
    delete_existing_collection()

    # Create the Chroma instance with the client and embedding function
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma
