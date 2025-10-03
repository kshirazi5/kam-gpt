import openai
from typing import List, Dict

def setup_openai(api_key: str):
    """Initialize OpenAI client with API key."""
    openai.api_key = api_key

def generate_chat_response(messages: List[Dict[str, str]], system_prompt: str = None) -> str:
    """Generate a response using GPT-4.1 through OpenAI's API."""
    try:
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
            
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",  # Using GPT-4.1
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"