"""
visualize.py — 2-D embedding space plots and loss curves.

High-dimensional embeddings (e.g. 50-D) are projected to 2-D using PCA
so we can visually confirm that semantically similar words cluster together.
For 2-D embeddings the projection is skipped.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from src.model import Word2Vec


# ── Color palette (matches README badge theme) ───────────────────────────────
PALETTE = {
    "bg":        "#0d1117",
    "grid":      "#21262d",
    "text":      "#e6edf3",
    "accent":    "#58a6ff",
    "highlight": "#f78166",
    "muted":     "#8b949e",
}


def _get_2d_embeddings(model: Word2Vec) -> np.ndarray:
    """Returns embeddings reduced to 2-D (PCA if needed)."""
    weights = model.embedding.weight.detach().numpy()   # (vocab, dim)

    if weights.shape[1] == 2:
        return weights

    # PCA to 2-D
    weights_centered = weights - weights.mean(axis=0)
    _, _, Vt = np.linalg.svd(weights_centered, full_matrices=False)
    return weights_centered @ Vt[:2].T


def plot_embeddings(model: Word2Vec, idx2word: dict, save_path: str = None) -> None:
    """
    Plots word embeddings in 2-D.

    Args:
        model (Word2Vec): Trained model.
        idx2word (dict): Index-to-word mapping.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    coords = _get_2d_embeddings(model)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"])

    # Scatter
    ax.scatter(
        coords[:, 0], coords[:, 1],
        color=PALETTE["accent"], s=90, zorder=3, edgecolors=PALETTE["bg"], linewidths=1.5,
    )

    # Labels
    for i, word in idx2word.items():
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
            color=PALETTE["text"],
            fontfamily="monospace",
            path_effects=[
                pe.withStroke(linewidth=2, foreground=PALETTE["bg"])
            ],
        )

    ax.set_title(
        "Word Embedding Space  ·  mini-embedding",
        color=PALETTE["text"], fontsize=13, pad=14, loc="left",
    )
    ax.set_xlabel("PC 1" if model.embedding.weight.shape[1] != 2 else "dim 0",
                  color=PALETTE["muted"])
    ax.set_ylabel("PC 2" if model.embedding.weight.shape[1] != 2 else "dim 1",
                  color=PALETTE["muted"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"Plot saved → {save_path}")

    plt.show()


def plot_loss(loss_history: list, save_path: str = None) -> None:
    """
    Plots the training loss curve.

    Args:
        loss_history (list): Per-epoch loss values returned by `train()`.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"])

    ax.plot(
        loss_history,
        color=PALETTE["highlight"], linewidth=2, zorder=3,
    )
    ax.set_title("Training Loss", color=PALETTE["text"], fontsize=13, pad=14, loc="left")
    ax.set_xlabel("Epoch", color=PALETTE["muted"])
    ax.set_ylabel("Cross-Entropy Loss", color=PALETTE["muted"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"Plot saved → {save_path}")

    plt.show()
