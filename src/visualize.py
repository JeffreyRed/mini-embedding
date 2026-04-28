"""
visualize.py — 2-D embedding plots, loss curves, training animation,
               and live matrix update inspection.

High-dimensional embeddings (e.g. 50-D) are projected to 2-D using PCA
so we can visually confirm that semantically similar words cluster together.
For 2-D embeddings the projection is skipped.

animate_training()   — requires EMBEDDING_DIM=2 and Pillow.
plot_matrix_update() — shows row-by-row gradient updates as a heatmap.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

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


def animate_training(
    snapshots: list,
    idx2word: dict,
    save_path: str = None,
    interval: int = 200,
) -> None:
    """
    Animates how word embeddings move through 2-D space during training.

    Each frame is one epoch snapshot captured during training.
    Words that share context gradually drift together; unrelated words drift apart.

    Requires EMBEDDING_DIM=2 (no PCA needed — coordinates are used directly).

    Args:
        snapshots (list): List of (epoch, weight_tensor) tuples from train().
        idx2word (dict): Index-to-word mapping.
        save_path (str, optional): If provided, saves the animation as a .gif.
        interval (int): Milliseconds between frames.
    """
    if not snapshots:
        print("No snapshots to animate. Set EMBEDDING_DIM=2 and pass snapshots= to train().")
        return

    _, sample = snapshots[0]
    if sample.shape[1] != 2:
        print("animate_training() requires EMBEDDING_DIM=2. Skipping animation.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"])

    # Compute global axis limits across all frames so the view stays stable
    all_coords = np.stack([w.numpy() for _, w in snapshots])  # (frames, vocab, 2)
    margin = 0.3
    ax.set_xlim(all_coords[:, :, 0].min() - margin, all_coords[:, :, 0].max() + margin)
    ax.set_ylim(all_coords[:, :, 1].min() - margin, all_coords[:, :, 1].max() + margin)

    scatter = ax.scatter([], [], color=PALETTE["accent"], s=90, zorder=3,
                         edgecolors=PALETTE["bg"], linewidths=1.5)
    labels  = [
        ax.text(0, 0, word, color=PALETTE["text"], fontsize=10,
                fontfamily="monospace", zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground=PALETTE["bg"])])
        for word in idx2word.values()
    ]
    title = ax.set_title("", color=PALETTE["text"], fontsize=12, loc="left", pad=12)

    def update(frame_idx):
        epoch, weights = snapshots[frame_idx]
        coords = weights.numpy()                    # (vocab, 2)
        scatter.set_offsets(coords)
        for i, txt in enumerate(labels):
            txt.set_position((coords[i, 0] + 0.03, coords[i, 1] + 0.03))
        title.set_text(
            f"Word Embedding Space  ·  epoch {epoch:>3}  "
            f"[{frame_idx + 1}/{len(snapshots)} frames]"
        )
        return [scatter, title] + labels

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=interval,
        blit=False,
    )

    if save_path:
        anim.save(save_path, writer="pillow", dpi=100)
        print(f"Animation saved → {save_path}")

    plt.tight_layout()
    plt.show()


def plot_matrix_updates(
    model: Word2Vec,
    dataset,
    idx2word: dict,
    n_steps: int = 6,
    save_path: str = None,
) -> None:
    """
    Runs `n_steps` training steps and plots the embedding matrix before
    and after each one, highlighting exactly which row changed and by how much.

    This makes the gradient-update mechanism fully visible:
      - The heatmap shows the full (vocab × dim) embedding matrix.
      - The changed row is highlighted in orange.
      - A delta panel shows the actual numeric change per dimension.
      - The (target, context) pair that caused the update is printed above.

    Args:
        model (Word2Vec): A freshly initialised (or partially trained) model.
        dataset: TextDataset — used to sample (target, context) pairs.
        idx2word (dict): Index-to-word mapping.
        n_steps (int): Number of update steps to visualise (default 6).
        save_path (str, optional): Save the figure to this path if provided.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import copy

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Sample the first n_steps pairs from the dataset
    pairs = list(dataset.get_batches())[:n_steps]

    vocab_size, emb_dim = model.embedding.weight.shape
    words = [idx2word[i] for i in range(vocab_size)]

    fig = plt.figure(figsize=(14, 3.5 * n_steps))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "Embedding Matrix Updates — one step at a time",
        color=PALETTE["text"], fontsize=13, y=1.001,
    )

    for step, (target_t, context_t) in enumerate(pairs):
        target_idx  = target_t.item()
        context_idx = context_t.item()
        target_word  = idx2word[target_idx]
        context_word = idx2word[context_idx]

        # Capture matrix BEFORE the update
        before = model.embedding.weight.detach().clone().numpy()

        # One training step
        optimizer.zero_grad()
        logits = model(target_t)
        loss   = loss_fn(logits, context_t)
        loss.backward()

        # Capture gradients for the target row before the step
        grad = model.embedding.weight.grad[target_idx].detach().clone().numpy()

        optimizer.step()

        # Capture matrix AFTER the update
        after = model.embedding.weight.detach().clone().numpy()
        delta = after - before   # everything else will be ~0

        # ── Plot row ───────────────────────────────────────────────────────
        gs  = GridSpec(1, 3, figure=fig,
                       top=1 - step / n_steps - 0.01,
                       bottom=1 - (step + 1) / n_steps + 0.02,
                       left=0.05, right=0.97,
                       wspace=0.35)

        # Panel 1 — matrix BEFORE
        ax1 = fig.add_subplot(gs[0])
        _draw_matrix(ax1, before, words, highlight_row=target_idx,
                     title=f"Step {step+1}  BEFORE\ntarget='{target_word}'  context='{context_word}'")

        # Panel 2 — matrix AFTER
        ax2 = fig.add_subplot(gs[1])
        _draw_matrix(ax2, after, words, highlight_row=target_idx,
                     title=f"Step {step+1}  AFTER\nloss={loss.item():.3f}")

        # Panel 3 — delta (what changed)
        ax3 = fig.add_subplot(gs[2])
        _draw_delta(ax3, delta, grad, words, target_idx,
                    title=f"Δ  (after − before)\nOnly row '{target_word}' moves")

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Matrix update plot saved → {save_path}")

    plt.show()


# ── Internal helpers for plot_matrix_updates ─────────────────────────────────

def _draw_matrix(ax, matrix: np.ndarray, words: list,
                 highlight_row: int, title: str) -> None:
    """Draws the embedding matrix as a heatmap with one row highlighted."""
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, color=PALETTE["text"], fontsize=8, pad=6)

    vmax = np.abs(matrix).max() or 1.0
    im   = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)

    # Highlight the updated row with an orange rectangle
    rect = plt.Rectangle(
        (-0.5, highlight_row - 0.5),
        matrix.shape[1], 1,
        linewidth=2, edgecolor="#f78166", facecolor="none", zorder=5,
    )
    ax.add_patch(rect)

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=7, color=PALETTE["text"],
                       fontfamily="monospace")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"d{i}" for i in range(matrix.shape[1])],
                       fontsize=7, color=PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    # Print numeric values inside cells
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            color = "#ffffff" if abs(matrix[r, c]) > vmax * 0.5 else PALETTE["muted"]
            ax.text(c, r, f"{matrix[r, c]:.3f}", ha="center", va="center",
                    fontsize=6, color=color)


def _draw_delta(ax, delta: np.ndarray, grad: np.ndarray,
                words: list, target_idx: int, title: str) -> None:
    """Draws only the changed row's delta and gradient values."""
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, color=PALETTE["text"], fontsize=8, pad=6)

    n_dims = delta.shape[1]
    x      = np.arange(n_dims)
    width  = 0.35

    changed = delta[target_idx]   # only this row has non-zero delta

    bars_delta = ax.bar(x - width / 2, changed, width,
                        color=PALETTE["highlight"], label="Δ weight", alpha=0.85)
    bars_grad  = ax.bar(x + width / 2, -grad, width,
                        color=PALETTE["accent"], label="−gradient", alpha=0.65)

    ax.axhline(0, color=PALETTE["muted"], linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d{i}" for i in x], fontsize=7, color=PALETTE["muted"])
    ax.tick_params(axis="y", colors=PALETTE["muted"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.set_facecolor(PALETTE["bg"])

    # Annotate bar values
    for bar in bars_delta:
        h = bar.get_height()
        if abs(h) > 1e-5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + np.sign(h) * 0.002,
                    f"{h:.4f}", ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=6, color=PALETTE["highlight"])

    ax.set_xlabel(f"dims of row '{words[target_idx]}'",
                  color=PALETTE["muted"], fontsize=7)
    legend = ax.legend(fontsize=7, loc="upper right",
                       facecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
