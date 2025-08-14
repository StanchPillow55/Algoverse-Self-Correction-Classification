"""
Embedding generator for creating semantic representations of text.

Supports various models including Sentence Transformers and OpenAI embeddings.
"""

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print("sentence-transformers not installed, local embedding models may not work.")

from typing import List, Dict
from ..utils.config import Config
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using different models."""

    def __init__(self, model_name: str = None):
        """Initialize the embedding generator with a specific model. Uses placeholders for external APIs."""
        Config.load()
        self.model_name = model_name or Config.DEFAULT_EMBEDDING_MODEL
        self.api_enabled = False  # Flag for API usage (to be enabled when keys are available)

        if self.model_name.startswith("openai") or self.model_name.startswith("llama"):
            self.api_enabled = True
            # Setup for future API embedding generation
            logger.warning("Using placeholder for API models. Please provide API keys.")
        else:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Initialized EmbeddingGenerator with local model {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize local model: {e}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts, using local models or API placeholders."""
        if self.api_enabled:
            embeddings = []
            for text in texts:
                # Placeholder logic for future real API call
                embeddings.append([0.0] * 384)  # Fake embedding size for placeholders
            logger.info("Generated placeholders for API-based embeddings.")
            return embeddings
        else:
            return self.model.encode(texts, convert_to_tensor=True).tolist()  # Use local models

    def get_default_model(self) -> str:
        """Get the default embedding model being used."""
        return self.model_name

