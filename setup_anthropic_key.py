#!/usr/bin/env python3
"""
Script to help set up the ANTHROPIC_API_KEY for ToolQA experiments.

This script helps configure the environment for running ToolQA experiments with Claude-Sonnet.
"""

import os
import sys

def check_api_key():
    """Check if ANTHROPIC_API_KEY is set."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print("✅ ANTHROPIC_API_KEY is already set")
        print(f"   Key starts with: {api_key[:8]}...")
        return True
    else:
        print("❌ ANTHROPIC_API_KEY is not set")
        return False

def setup_instructions():
    """Print setup instructions."""
    print("""
🔧 To set up your Anthropic API key for the ToolQA experiment:

Option 1: Set it for this session
  export ANTHROPIC_API_KEY="your-api-key-here"

Option 2: Set it permanently (add to ~/.zshrc or ~/.bashrc)
  echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.zshrc
  source ~/.zshrc

Option 3: Create a .env file (if using python-dotenv)
  echo 'ANTHROPIC_API_KEY=your-api-key-here' > .env

📋 You can get your API key from: https://console.anthropic.com/

Once set, run this script again to verify, then execute the ToolQA experiment:
  python3 run_anthropic_multiturn_experiments.py --models claude-3-5-sonnet-20241210 --datasets toolqa_deterministic_100 --sample_sizes 100
""")

def test_anthropic_connection():
    """Test if we can connect to Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic()  # Will use ANTHROPIC_API_KEY from env
        
        # Try a simple test call
        print("🧪 Testing Anthropic API connection...")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Use correct model ID
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("✅ Anthropic API connection successful!")
        print(f"   Response: {response.content[0].text}")
        return True
        
    except ImportError:
        print("❌ anthropic library not installed. Run: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ Anthropic API connection failed: {e}")
        return False

def main():
    """Main function."""
    print("🔍 Checking Anthropic API setup for ToolQA experiments...")
    
    # Check if API key is set
    if check_api_key():
        # Test the connection
        if test_anthropic_connection():
            print("\n✅ All set! You can now run the ToolQA experiment:")
            print("   python3 run_anthropic_multiturn_experiments.py --models claude-3-5-sonnet-20241210 --datasets toolqa_deterministic_100 --sample_sizes 100")
        else:
            print("\n⚠️ API key is set but connection failed. Please check your key.")
    else:
        setup_instructions()

if __name__ == "__main__":
    main()