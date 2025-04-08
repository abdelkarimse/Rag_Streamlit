import sqlite3
import logging
import os
from typing import List, Dict, Optional
from utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

def get_db_connection():
    """Get a database connection to SQLite."""
    conn = sqlite3.connect("chat_sessions/chat_history.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def save_text_message(session_key: str, sender_type: str, text_content: str, user_id: int) -> bool:
    """
    Save a text message to the messages table.
    
    Args:
        session_key: The session identifier
        sender_type: Type of sender (e.g., 'user', 'assistant')
        text_content: The text content of the message
        user_id: The ID of the user
    
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    try:
        session_key = session_key.strip()
        cursor = conn.cursor()
        
        # Get session ID or create if doesn't exist
        cursor.execute(
            "SELECT session_id FROM chat_sessions WHERE session_key = ? AND user_id = ?", 
            (session_key, user_id)
        )
        session_result = cursor.fetchone()
        
        if session_result is None:
            # Create a new session
            cursor.execute(
                "INSERT INTO chat_sessions (session_key, user_id) VALUES (?, ?)",
                (session_key, user_id)
            )
            conn.commit()
            
            # Get the new session ID
            cursor.execute(
                "SELECT session_id FROM chat_sessions WHERE session_key = ? AND user_id = ?", 
                (session_key, user_id)
            )
            session_result = cursor.fetchone()
            
        chat_history_id = session_result['session_id']
        
        # Insert the message
        cursor.execute(
            'INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
            (chat_history_id, sender_type, 'text', text_content)
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving text message: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def load_messages(chat_history_id: str, user_id: int) -> List[Dict]:
    """
    Load all messages for a given chat history ID and user ID.
    
    Args:
        chat_history_id: The chat history identifier
        user_id: The ID of the user
    
    Returns:
        List[Dict]: List of message dictionaries
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Get the session ID
        cursor.execute(
            "SELECT session_id FROM chat_sessions WHERE session_key = ? AND user_id = ?",
            (chat_history_id, user_id)
        )
        session_result = cursor.fetchone()
        
        if session_result is None:
            logger.warning(f"No session found for chat_history_id: {chat_history_id} and user_id: {user_id}")
            return []

        # Get all messages for the session
        cursor.execute(
            "SELECT message_id, sender_type, message_type, text_content, blob_content FROM messages WHERE chat_history_id = ?",
            (session_result['session_id'],)
        )
        messages = cursor.fetchall()
        
        chat_history = []
        for message in messages:
            chat_history.append({
                'message_id': message['message_id'],
                'sender_type': message['sender_type'],
                'message_type': message['message_type'],
                'content': message['text_content'] if message['message_type'] == 'text' else message['blob_content']
            })

        return chat_history
    except Exception as e:
        logger.error(f"Error loading messages: {e}")
        return []
    finally:
        conn.close()

def load_last_k_text_messages_ollama(chat_history_id: str, k: int, user_id: int = 1) -> List[Dict]:
    """
    Load the last k text messages for a given chat history ID and convert to Ollama format.
    
    Args:
        chat_history_id: The chat history identifier
        k: Number of messages to retrieve
        user_id: The ID of the user
    
    Returns:
        List[Dict]: List of message dictionaries in the format required for Ollama API
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Get the session ID
        cursor.execute(
            "SELECT session_id FROM chat_sessions WHERE session_key = ? AND user_id = ?",
            (chat_history_id, user_id)
        )
        session_result = cursor.fetchone()
        
        if session_result is None:
            # No session found, return empty list
            return []
            
        # Get the last k text messages
        cursor.execute(
            """
            SELECT message_id, sender_type, message_type, text_content
            FROM messages
            WHERE chat_history_id = ? AND message_type = 'text'
            ORDER BY message_id DESC
            LIMIT ?
            """,
            (session_result['session_id'], k)
        )
        messages = cursor.fetchall()
        
        # Format messages for Ollama
        chat_history = []
        for message in reversed(messages):  # Reverse to get correct chronological order
            chat_history.append({
                'sender_type': message['sender_type'],
                'content': message['text_content']
            })
            
        return chat_history
    except Exception as e:
        logger.error(f"Error loading text messages for Ollama: {e}")
        return []
    finally:
        conn.close()

def get_all_chat_history_ids(user_id: int = 1) -> List[str]:
    """
    Get all chat history IDs for a user.
    
    Args:
        user_id: The ID of the user
    
    Returns:
        List[str]: List of chat history IDs
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_key FROM chat_sessions WHERE user_id = ? ORDER BY session_id DESC",
            (user_id,)
        )
        results = cursor.fetchall()
        return [result['session_key'] for result in results]
    except Exception as e:
        logger.error(f"Error getting chat history IDs: {e}")
        return []
    finally:
        conn.close()

def delete_chat_history(chat_history_id: str, user_id: int) -> bool:
    """
    Delete a chat history.
    
    Args:
        chat_history_id: The chat history identifier
        user_id: The ID of the user
    
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Get the session ID
        cursor.execute(
            "SELECT session_id FROM chat_sessions WHERE session_key = ? AND user_id = ?",
            (chat_history_id, user_id)
        )
        session_result = cursor.fetchone()
        
        if session_result is None:
            logger.warning(f"No session found for chat_history_id: {chat_history_id} and user_id: {user_id}")
            return False
        
        # Delete messages first (foreign key constraint)
        cursor.execute(
            "DELETE FROM messages WHERE chat_history_id = ?",
            (session_result['session_id'],)
        )
        
        # Delete the session
        cursor.execute(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            (session_result['session_id'],)
        )
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting chat history: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def init_db() -> None:
    """Initialize the database schema."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create chat_sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_key, user_id)
        )
        ''')
        
        # Create messages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_history_id INTEGER NOT NULL,
            sender_type TEXT NOT NULL,
            message_type TEXT NOT NULL,
            text_content TEXT,
            blob_content BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_history_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
        )
        ''')
        
        # Create users table for future authentication
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()
