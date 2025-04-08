from datetime import datetime
import base64
import yaml
import requests
from dotenv import load_dotenv
import streamlit as st
import os
import aiohttp
import asyncio
import time

# Load environment variables
load_dotenv()

def convert_ns_to_seconds(ns_value):


    """Convert nanoseconds to seconds."""
    return ns_value / 1_000_000_000 

def timeit(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        wrapper: Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def command(user_input):
    """Parse user commands."""
    splitted_input = user_input.split(" ")
    if splitted_input[0] == "/pull" and len(splitted_input) > 1:
        return pull_model_in_background(splitted_input[1])
    elif splitted_input[0] == "/list":
        return list_ollama_models()
    elif splitted_input[0] == "/help":
        return "الأوامر الممكنة:\n- /pull <اسم_النموذج>\n- /list\n- /help"
    else:    
        return """أمر غير صالح، يرجى استخدام أحد الأوامر التالية:\n
                    - /help\n
                    - /pull <اسم_النموذج>\n
                    - /list"""

def pull_ollama_model(model_name):
    """Pull an Ollama model synchronously."""
    try:
        json_response = requests.post(url="http://127.0.0.1:11434/api/pull", json={"model": model_name}).json()
        print(json_response)
        if "error" in json_response:
            return json_response["error"]["message"]
        else:
            st.session_state.model_options = list_ollama_models()
            st.warning(f"Pulling {model_name} finished.")
            return json_response
    except requests.exceptions.RequestException as e:
        st.error(f"Error pulling model: {str(e)}")
        return None

async def pull_ollama_model_async(model_name, stream=False, retries=3):
    """
    Pull an Ollama model asynchronously.
    
    Args:
        model_name: Name of the model to pull
        stream: Whether to stream the response
        retries: Number of retries
        
    Returns:
        str: Result message
    """
    url = "http://127.0.0.1:11434/api/pull"
    json_data = {"model": model_name, "stream": stream}
    
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
                async with session.post(url, json=json_data) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        if "error" in json_response:
                            return f"Error: {json_response['error']}"
                        return f"Successfully pulled {model_name}"
                    else:
                        error_text = await response.text()
                        return f"Error: {response.status} - {error_text}"
        except Exception as e:
            if attempt == retries - 1:
                return f"Failed to pull {model_name}: {str(e)}"
            print(f"Attempt {attempt + 1} failed: {str(e)}")
    
    return f"Failed to pull {model_name} after {retries} attempts"

def pull_model_in_background(model_name, stream=False):
    """Trigger the async pull in the background."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # If no loop is running, start a new one
        loop = None

    if loop and loop.is_running():
        # If an event loop is already running, create a task for the async function
        return asyncio.create_task(pull_ollama_model_async(model_name, stream=stream))
    else:
        # Otherwise, use asyncio.run() to run it synchronously
        return asyncio.run(pull_ollama_model_async(model_name, stream=stream))

def list_openai_models():
    """List available OpenAI models."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    response = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {openai_api_key}"}).json()
    if response.get("error", False):
        # st.warning("OpenAI Error: " + response["error"]["message"])
        return []
    else:
        return [item["id"] for item in response["data"]]

def list_ollama_models():
    """
    List available Ollama models.
    
    Returns:
        list: List of available model names
    """
    try:
        response = requests.get(url="http://127.0.0.1:11434/api/tags", timeout=5)
        response.raise_for_status()
        json_response = response.json()
        
        if json_response.get("error", False):
            return []
        
        # Filter out embedding models
        models = [model["name"] for model in json_response["models"] if "embed" not in model["name"]]

        return models
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Ollama models: {str(e)}")
        return ["bge-m3:latest"]  # Default model if can't fetch list

def load_config(file_path="config.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def convert_bytes_to_base64(image_bytes):
    """Convert bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")
    
def convert_bytes_to_base64_with_prefix(image_bytes):
    """Convert bytes to base64 string with image prefix."""
    return "data:image/jpeg;base64," + convert_bytes_to_base64(image_bytes)

def get_timestamp():
    """Get the current timestamp in a specific format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_avatar(sender_type):
    """
    Get the avatar image path based on the sender type.
    
    Args:
        sender_type: Type of sender ("user" or "assistant")
        
    Returns:
        str: Path to avatar image
    """
    if sender_type == "user":
        return "chat_icons/user_image.png"
    else:
        return "chat_icons/bot_image.png"
