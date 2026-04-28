"""
train.py — Training loop for the Word2Vec model.

Uses cross-entropy loss: for each target word, the model predicts which
vocabulary words are likely context words. The embedding matrix is updated
via backpropagation after every pair, gradually encoding semantic structure
into the vector space.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import TextDataset
from src.model import Word2Vec


def train(
    model: Word2Vec,
    dataset: TextDataset,
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
    snapshots: list = None,
    snapshot_every: int = 5,
) -> list:
    """
    Train the Word2Vec model using Skip-gram with cross-entropy loss.

    Args:
        model (Word2Vec): The model to train.
        dataset (TextDataset): Training data with (target, context) pairs.
        epochs (int): Number of full passes over the training data.
        lr (float): Learning rate for the Adam optimizer.
        verbose (bool): If True, prints loss every 10 epochs.
        snapshots (list): If provided, embedding matrix copies are appended
            every `snapshot_every` epochs. Used for training animation.
        snapshot_every (int): How often (in epochs) to save a snapshot.

    Returns:
        List of per-epoch total losses (useful for plotting a loss curve).
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for target, context in dataset.get_batches():
            optimizer.zero_grad()

            logits = model(target)            # (1, vocab_size)
            loss   = loss_fn(logits, context) # scalar

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_history.append(total_loss)

        # Save a copy of the embedding weights for animation
        if snapshots is not None and (epoch % snapshot_every == 0 or epoch == 1):
            snapshots.append((epoch, model.embedding.weight.detach().clone()))

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch [{epoch:>3}/{epochs}]  Loss: {total_loss:.4f}")

    return loss_history
