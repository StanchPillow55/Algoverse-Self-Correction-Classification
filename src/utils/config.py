"""
Configuration management for the LLM error classification pipeline.

Loads settings from a .env file and provides access to configuration parameters.
"""

import os
from dotenv import load_dotenv

class Config:
    """Configuration class to load environment variables."""

    _loaded = False

    OPENAI_API_KEY: str = None
    ANTHROPIC_API_KEY: str = None
    DEFAULT_EMBEDDING_MODEL: str = None
    OPENAI_EMBEDDING_MODEL: str = None
    DATA_DIR: str = None
    MODEL_DIR: str = None

    @classmethod
    def load(cls, env_file: str = '.env'):
        """Load environment variables from a file."""
        if not cls._loaded:
            load_dotenv(env_file)
            cls.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            cls.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
            cls.DEFAULT_EMBEDDING_MODEL = os.getenv('DEFAULT_EMBEDDING_MODEL')
            cls.OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL')
            cls.DATA_DIR = os.getenv('DATA_DIR')
            cls.MODEL_DIR = os.getenv('MODEL_DIR')
            cls._loaded = True

    @staticmethod
    def get(key: str) -> str:
        """Get a configuration value by key."""
        return os.getenv(key)

