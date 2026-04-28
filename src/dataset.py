"""
dataset.py — Tokenization and Skip-gram pair generation.

Reads a plain-text corpus, builds a vocabulary, and produces
(target, context) index pairs used to train the embedding model.
"""

import torch
from collections import Counter
from typing import List, Tuple


class TextDataset:
    """
    Tokenizes a corpus and generates Skip-gram training pairs.

    Args:
        path (str): Path to the plain-text corpus file.
        window_size (int): Number of context words on each side of the target.

    Attributes:
        word2idx (dict): Maps each word to a unique integer index.
        idx2word (dict): Reverse mapping from index to word.
        data (list): List of (target_idx, context_idx) training pairs.
        vocab_size (int): Number of unique tokens in the corpus.
    """

    def __init__(self, path: str, window_size: int = 2) -> None:
        self.window_size = window_size

        with open(path, "r") as f:
            self.sentences: List[List[str]] = [
                line.strip().split() for line in f if line.strip()
            ]

        self._build_vocab()
        self.data: List[Tuple[int, int]] = self._build_training_pairs()
        self.vocab_size: int = len(self.word2idx)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_vocab(self) -> None:
        """Creates word↔index mappings from the corpus."""
        words = [w for sentence in self.sentences for w in sentence]
        vocab = sorted(set(words))
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def _build_training_pairs(self) -> List[Tuple[int, int]]:
        """Generates (target, context) pairs using a sliding window."""
        pairs: List[Tuple[int, int]] = []
        for sentence in self.sentences:
            n = len(sentence)
            for i, word in enumerate(sentence):
                target_idx = self.word2idx[word]
                for offset in range(-self.window_size, self.window_size + 1):
                    if offset == 0:
                        continue
                    j = i + offset
                    if 0 <= j < n:
                        context_idx = self.word2idx[sentence[j]]
                        pairs.append((target_idx, context_idx))
        return pairs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batches(self):
        """Yields one (target, context) tensor pair at a time."""
        for target, context in self.data:
            yield torch.tensor([target]), torch.tensor([context])

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"TextDataset(vocab_size={self.vocab_size}, "
            f"pairs={len(self.data)}, window_size={self.window_size})"
        )
