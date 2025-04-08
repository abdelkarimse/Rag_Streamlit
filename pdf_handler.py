import os
from utils import load_config, timeit
from vectordb_handler import load_vectordb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import pypdfium2

# Load configuration
config = load_config()

def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from a PDF using pypdfium2
    
    Args:
        pdf_bytes: Bytes of the PDF file
        
    Returns:
        str: Extracted text
    """
    try:
        pdf_file = pypdfium2.PdfDocument(pdf_bytes)
        text = "\n".join(
            pdf_file.get_page(page_number).get_textpage().get_text_range() 
            for page_number in range(len(pdf_file))
        )
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def get_pdf_texts(pdfs_bytes_list):
    """
    Extract text from a list of PDF files
    
    Args:
        pdfs_bytes_list: List of PDF file bytes
        
    Returns:
        list: List of extracted texts
    """
    return [extract_text_from_pdf(pdf_bytes) for pdf_bytes in pdfs_bytes_list]

def get_text_chunks(text):
    """
    Split text into chunks using RecursiveCharacterTextSplitter
    
    Args:
        text: Text to split
        
    Returns:
        list: List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["pdf_text_splitter"]["chunk_size"], 
        chunk_overlap=config["pdf_text_splitter"]["overlap"],
        separators=config["pdf_text_splitter"]["separators"]
    )
    return splitter.split_text(text)

def get_document_chunks(text_list):
    """
    Convert text chunks to document objects
    
    Args:
        text_list: List of texts to convert
        
    Returns:
        list: List of document objects
    """
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content=chunk))
    return documents

@timeit
def add_documents_to_db(pdfs_bytes):
    """
    Add documents to vector database
    
    Args:
        pdfs_bytes: List of PDF file objects from Streamlit uploader
        
    Returns:
        None
    """
    try:
        # Get PDF bytes from Streamlit UploadedFile objects
        pdf_bytes = [pdf.read() for pdf in pdfs_bytes]
        
        # Extract text from PDFs
        texts = get_pdf_texts(pdf_bytes)
        
        # Convert to document chunks
        documents = get_document_chunks(texts)
        
        # Load vector database and add documents
        vector_db = load_vectordb()
        vector_db.add_documents(documents)
        
        print(f"Added {len(documents)} document chunks to vector database")
    except Exception as e:
        print(f"Error adding documents to database: {e}")

def process_pdf_folder(folder_path):
    """
    Process all PDFs in a folder
    
    Args:
        folder_path: Path to folder containing PDFs
        
    Returns:
        None
    """
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                    add_documents_to_db([pdf_bytes])
        print(f"Processed all PDFs in {folder_path}")
    except Exception as e:
        print(f"Error processing PDF folder: {e}")