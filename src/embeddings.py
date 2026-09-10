"""
Generic embedding wrapper that works uniformly across sentence-transformers
models AND plain HuggingFace checkpoints (like legal-BERT) via mean pooling.
Using the same pooling strategy for all three models keeps the comparison
fair -- differences in results reflect the models themselves, not
inconsistent pooling logic.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Loads a model as a SentenceTransformer. For models already built for
    sentence embeddings (all-MiniLM-L6-v2, bge-small) this just works.
    For plain HF checkpoints (legal-bert) that have no built-in pooling
    head, we construct one manually: transformer layer + mean pooling.
    """
    try:
        # works directly for models published as proper sentence-transformers
        return SentenceTransformer(model_name)
    except Exception:
        # fallback: build transformer + mean-pooling manually for plain
        # HF checkpoints like nlpaueb/legal-bert-base-uncased
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Returns L2-normalized embeddings -- normalizing means we can use
    inner product in FAISS (IndexFlatIP) to get cosine similarity, which
    is faster than computing true cosine distance directly."""
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms