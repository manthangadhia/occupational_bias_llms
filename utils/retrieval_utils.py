from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

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

# TODO: add query function to return top k results