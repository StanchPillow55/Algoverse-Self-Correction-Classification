"""
Together AI provider implementation for LearnerBot.
Compatible with OpenAI-style API interface.
"""

import os
import requests
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TogetherResponse:
    """Response from Together AI API"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str

class TogetherAIProvider:
    """Together AI provider for Llama models"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.together.xyz/v1"):
        """
        Initialize Together AI provider
        
        Args:
            api_key: Together AI API key (or from env var TOGETHER_API_KEY)
            base_url: Together AI API base URL
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("Together AI API key not found. Set TOGETHER_API_KEY environment variable.")
    
    def _make_request(self, messages: List[Dict[str, str]], model: str, 
                     temperature: float = 0.0, max_tokens: int = 4000,
                     retries: int = 3) -> TogetherResponse:
        """Make API request to Together AI"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return TogetherResponse(
                        content=result["choices"][0]["message"]["content"],
                        usage=result.get("usage", {}),
                        model=result.get("model", model),
                        finish_reason=result["choices"][0].get("finish_reason", "stop")
                    )
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"API error {response.status_code}: {response.text}")
                    if attempt == retries - 1:
                        raise Exception(f"API call failed: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded")
    
    def generate_response(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """Generate response using Together AI (compatible with LearnerBot interface)"""
        
        response = self._make_request(messages, model, **kwargs)
        return response.content
    
    def get_available_models(self) -> List[str]:
        """Get list of available Llama models on Together AI"""
        return [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf", 
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3-8b-chat-hf",
            "meta-llama/Llama-3-70b-chat-hf"
        ]

# Integration point for LearnerBot
def create_together_client(model_config: Dict[str, Any]) -> TogetherAIProvider:
    """Create Together AI client from model config"""
    return TogetherAIProvider()

def test_together_connection():
    """Test Together AI connection"""
    try:
        provider = TogetherAIProvider()
        
        test_messages = [
            {"role": "user", "content": "What is 2+2? Give a very brief answer."}
        ]
        
        response = provider.generate_response(
            messages=test_messages,
            model="meta-llama/Llama-2-7b-chat-hf",
            temperature=0.0,
            max_tokens=50
        )
        
        print("✅ Together AI connection successful!")
        print(f"Test response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Together AI connection failed: {e}")
        return False

if __name__ == "__main__":
    # Test the connection
    test_together_connection()