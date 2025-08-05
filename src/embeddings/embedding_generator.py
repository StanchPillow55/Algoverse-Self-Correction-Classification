"""
Embedding generator for creating semantic representations of text.

Supports various models including Sentence Transformers and OpenAI embeddings.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
from ..utils.config import Config
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using different models."""

    def __init__(self, model_name: str = None):
        """Initialize the embedding generator with a specific model."""
        Config.load()
        self.model_name = model_name or Config.DEFAULT_EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Initialized EmbeddingGenerator with model {self.model_name}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def get_default_model(self) -> str:
        """Get the default embedding model being used."""
        return self.model_name

