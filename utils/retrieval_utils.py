from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import faiss

MODEL_NAME = "intfloat/e5-large-v2"
MAX_LENGTH = 512

def load_model(device="cuda", cache_dir=None):
    model = SentenceTransformer(MODEL_NAME, device=device, cache_folder=cache_dir)
    return model  # can access tokenizer via model.tokenizer

def embed_batch(texts, model, prefix="passage: ", **kwargs):
    prefixed = [f"{prefix}{t}" for t in texts]
    return model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

def query(query_text, model, index: faiss.IndexFlatIP, k: int = 10, prefix = "query: ") -> tuple[np.ndarray, np.ndarray]:
    """
    Embed a query string and retrieve top-k results from the FAISS index.

    Returns:
        distances: shape (k,) — inner product scores, higher is more similar
        faiss_ids: shape (k,) — positions in the index, use to look up metadata
    """
    emb = embed_batch([query_text], model, prefix=prefix)
    distances, faiss_ids = index.search(emb, k)
    return distances[0], faiss_ids[0]