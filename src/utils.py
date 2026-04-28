"""
utils.py — Cosine similarity and nearest-neighbor search.

After training, the embedding matrix encodes semantic relationships as
geometry: similar words end up close together in vector space.
Cosine similarity measures this closeness (1.0 = identical direction,
0.0 = orthogonal, -1.0 = opposite).
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple

from src.model import Word2Vec


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Computes cosine similarity between two 1-D vectors.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        Similarity score in [-1, 1].
    """
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


def most_similar(
    model: Word2Vec,
    word: str,
    word2idx: dict,
    idx2word: dict,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Returns the top-k words most similar to `word` by cosine similarity.

    Args:
        model (Word2Vec): Trained model.
        word (str): Query word (must be in vocabulary).
        word2idx (dict): Word-to-index mapping.
        idx2word (dict): Index-to-word mapping.
        top_k (int): Number of similar words to return.

    Returns:
        List of (word, similarity) tuples, sorted descending.

    Raises:
        KeyError: If `word` is not in the vocabulary.
    """
    if word not in word2idx:
        raise KeyError(f"'{word}' not found in vocabulary: {list(word2idx.keys())}")

    target_idx = word2idx[word]
    target_vec = model.get_embedding(target_idx)

    similarities: List[Tuple[str, float]] = []
    for idx in range(len(word2idx)):
        if idx == target_idx:
            continue
        sim = cosine_similarity(target_vec, model.get_embedding(idx))
        similarities.append((idx2word[idx], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def print_similar(
    model: Word2Vec,
    words: List[str],
    word2idx: dict,
    idx2word: dict,
    top_k: int = 3,
) -> None:
    """Pretty-prints nearest neighbors for a list of query words."""
    print("\n── Nearest Neighbors ──────────────────────────────")
    for word in words:
        try:
            results = most_similar(model, word, word2idx, idx2word, top_k)
            formatted = ", ".join(f"{w} ({s:.3f})" for w, s in results)
            print(f"  {word:>10}  →  {formatted}")
        except KeyError as e:
            print(f"  {word:>10}  →  {e}")
    print("────────────────────────────────────────────────────\n")
