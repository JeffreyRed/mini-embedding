"""
model.py — Skip-gram Word2Vec architecture.

Architecture:
    word index  →  Embedding lookup  →  Linear projection  →  vocab logits

The embedding layer is the core component: a trainable (vocab_size × embedding_dim)
matrix. The linear layer projects embeddings back to vocabulary size so the model
can be trained to predict context words via cross-entropy loss.

After training, only the embedding layer weights are used as word representations.
"""

import torch
import torch.nn as nn


class Word2Vec(nn.Module):
    """
    Minimal Skip-gram Word2Vec model.

    Args:
        vocab_size (int): Number of unique tokens in the vocabulary.
        embedding_dim (int): Dimensionality of the learned word vectors.

    Example::

        model = Word2Vec(vocab_size=20, embedding_dim=8)
        logits = model(torch.tensor([3]))   # shape: (1, 20)
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # Kaiming init for the linear layer for stable training
        nn.init.kaiming_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: LongTensor of shape (batch,) containing word indices.

        Returns:
            Logits tensor of shape (batch, vocab_size).
        """
        embedded = self.embedding(x)   # (batch, embedding_dim)
        logits = self.linear(embedded) # (batch, vocab_size)
        return logits

    def get_embedding(self, word_idx: int) -> torch.Tensor:
        """Returns the embedding vector for a single word index."""
        with torch.no_grad():
            return self.embedding.weight[word_idx]

    def __repr__(self) -> str:
        vocab_size, embedding_dim = self.embedding.weight.shape
        return (
            f"Word2Vec(vocab_size={vocab_size}, embedding_dim={embedding_dim})"
        )
