from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BOX_FACE = "#f7f3e8"
BOX_EDGE = "#2f3e46"
TITLE_COLOR = "#1f2d3d"
TEXT_COLOR = "#203040"
ACCENT = "#9d6b53"
BG = "#fffdf8"


def _add_box(ax, x: float, y: float, w: float, h: float, title: str, body: str, face: str = BOX_FACE) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=BOX_EDGE,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=12, weight="bold", color=TITLE_COLOR)
    ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=10, color=TEXT_COLOR, wrap=True)


def _add_arrow(ax, start: tuple[float, float], end: tuple[float, float], text: str | None = None) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color=ACCENT,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    if text:
        tx = (start[0] + end[0]) / 2
        ty = (start[1] + end[1]) / 2 + 0.03
        ax.text(tx, ty, text, ha="center", va="bottom", fontsize=9, color=ACCENT)


def _setup_canvas(title: str):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=22, weight="bold", color=TITLE_COLOR)
    return fig, ax


def draw_text_baseline(output_path: Path) -> None:
    fig, ax = _setup_canvas("Text Baseline")
    _add_box(ax, 0.06, 0.34, 0.18, 0.22, "Input", "source text\n+\nobserved replies\n(concatenated)")
    _add_box(ax, 0.31, 0.34, 0.20, 0.22, "HashingTextEncoder", "EmbeddingBag over\nstable hashed tokens\nmean pooling")
    _add_box(ax, 0.58, 0.34, 0.16, 0.22, "Shared MLP", "LayerNorm\nLinear\nReLU\nDropout")
    _add_box(ax, 0.80, 0.50, 0.14, 0.14, "Regression Head", "future_neg_ratio")
    _add_box(ax, 0.80, 0.22, 0.14, 0.14, "Classification Head", "future majority\nsentiment logits")
    _add_arrow(ax, (0.24, 0.45), (0.31, 0.45), "merged text")
    _add_arrow(ax, (0.51, 0.45), (0.58, 0.45), "hidden")
    _add_arrow(ax, (0.74, 0.45), (0.80, 0.57))
    _add_arrow(ax, (0.74, 0.45), (0.80, 0.29))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_temporal_baseline(output_path: Path) -> None:
    fig, ax = _setup_canvas("Temporal Baseline")
    _add_box(ax, 0.05, 0.34, 0.18, 0.22, "Input", "observed replies only\nordered as a sequence")
    _add_box(ax, 0.29, 0.34, 0.18, 0.22, "Reply Encoder", "HashingTextEncoder\nper reply")
    _add_box(ax, 0.53, 0.34, 0.18, 0.22, "Temporal Encoder", "packed LSTM\nlast hidden state")
    _add_box(ax, 0.75, 0.34, 0.14, 0.22, "Shared MLP", "LayerNorm\nLinear\nReLU\nDropout")
    _add_box(ax, 0.80, 0.62, 0.15, 0.12, "Regression Head", "future_neg_ratio")
    _add_box(ax, 0.80, 0.16, 0.15, 0.12, "Classification Head", "future majority\nsentiment logits")
    _add_arrow(ax, (0.23, 0.45), (0.29, 0.45), "reply texts")
    _add_arrow(ax, (0.47, 0.45), (0.53, 0.45), "reply embeddings")
    _add_arrow(ax, (0.71, 0.45), (0.75, 0.45), "sequence summary")
    _add_arrow(ax, (0.82, 0.56), (0.85, 0.62))
    _add_arrow(ax, (0.82, 0.34), (0.85, 0.28))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_structure_baseline(output_path: Path) -> None:
    fig, ax = _setup_canvas("Structure Baseline")
    _add_box(ax, 0.04, 0.56, 0.18, 0.18, "Text Input", "source text\n+\nobserved replies")
    _add_box(ax, 0.04, 0.24, 0.18, 0.18, "Tree Input", "thread id\nconversation tree\nobserved reply nodes")
    _add_box(ax, 0.29, 0.56, 0.19, 0.18, "Text Encoder", "HashingTextEncoder\non pooled thread text")
    _add_box(ax, 0.29, 0.24, 0.19, 0.18, "Tree Statistics", "8 structural features\nfrom compute_tree_statistics")
    _add_box(ax, 0.54, 0.24, 0.15, 0.18, "Tree Projection", "Linear(8 -> hidden)\nReLU\nDropout")
    _add_box(ax, 0.54, 0.56, 0.15, 0.18, "Fusion", "concat(text, tree)\nLayerNorm\nLinear\nReLU")
    _add_box(ax, 0.78, 0.60, 0.16, 0.12, "Regression Head", "future_neg_ratio")
    _add_box(ax, 0.78, 0.28, 0.16, 0.12, "Classification Head", "future majority\nsentiment logits")
    _add_arrow(ax, (0.22, 0.65), (0.29, 0.65))
    _add_arrow(ax, (0.22, 0.33), (0.29, 0.33))
    _add_arrow(ax, (0.48, 0.33), (0.54, 0.33), "8-d stats")
    _add_arrow(ax, (0.48, 0.65), (0.54, 0.65), "text embedding")
    _add_arrow(ax, (0.69, 0.60), (0.78, 0.66))
    _add_arrow(ax, (0.69, 0.60), (0.78, 0.34))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_affect_state_forecaster(output_path: Path) -> None:
    fig, ax = _setup_canvas("Affect State Forecaster")
    _add_box(ax, 0.03, 0.68, 0.16, 0.16, "Source Input", "source text")
    _add_box(ax, 0.03, 0.42, 0.16, 0.16, "Reply Input", "observed replies\nordered sequence")
    _add_box(ax, 0.03, 0.16, 0.16, 0.16, "Tree Input", "thread id\nconversation tree\nobserved replies")
    _add_box(ax, 0.25, 0.68, 0.16, 0.16, "Source Encoder", "HashingTextEncoder")
    _add_box(ax, 0.25, 0.42, 0.16, 0.16, "Reply Encoder", "HashingTextEncoder\nper reply")
    _add_box(ax, 0.25, 0.16, 0.16, 0.16, "Tree Statistics", "compute_tree_statistics\n8 structural features")
    _add_box(ax, 0.47, 0.42, 0.16, 0.16, "Temporal Encoder", "packed LSTM\nlast hidden state")
    _add_box(ax, 0.47, 0.16, 0.16, 0.16, "Tree Projection", "Linear(8 -> hidden)\nReLU\nDropout")
    _add_box(ax, 0.47, 0.68, 0.16, 0.16, "Source Embedding", "hidden text vector")
    _add_box(ax, 0.67, 0.42, 0.15, 0.24, "Fusion", "concat(source,\ntemporal,\nstructure)\nLayerNorm\nLinear\nGELU")
    _add_box(ax, 0.84, 0.54, 0.13, 0.16, "Affect-State Head", "latent current\ngroup affect state")
    _add_box(ax, 0.84, 0.77, 0.13, 0.10, "Regression Head", "future_neg_ratio")
    _add_box(ax, 0.84, 0.33, 0.13, 0.10, "Classification Head", "future majority\nsentiment logits")
    _add_box(ax, 0.66, 0.08, 0.16, 0.12, "Ablation Path", "can disable\ntemporal / structure /\naffect-state bottleneck", face="#efe5da")
    _add_arrow(ax, (0.19, 0.76), (0.25, 0.76))
    _add_arrow(ax, (0.19, 0.50), (0.25, 0.50))
    _add_arrow(ax, (0.19, 0.24), (0.25, 0.24))
    _add_arrow(ax, (0.41, 0.50), (0.47, 0.50))
    _add_arrow(ax, (0.41, 0.24), (0.47, 0.24))
    _add_arrow(ax, (0.41, 0.76), (0.47, 0.76))
    _add_arrow(ax, (0.63, 0.76), (0.67, 0.58))
    _add_arrow(ax, (0.63, 0.50), (0.67, 0.54))
    _add_arrow(ax, (0.63, 0.24), (0.67, 0.50))
    _add_arrow(ax, (0.82, 0.58), (0.84, 0.62), "latent state")
    _add_arrow(ax, (0.905, 0.70), (0.905, 0.77))
    _add_arrow(ax, (0.905, 0.54), (0.905, 0.43))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG architecture diagrams for ASF models.")
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_text_baseline(args.output_dir / "text_baseline_architecture.png")
    draw_temporal_baseline(args.output_dir / "temporal_baseline_architecture.png")
    draw_structure_baseline(args.output_dir / "structure_baseline_architecture.png")
    draw_affect_state_forecaster(args.output_dir / "affect_state_forecaster_architecture.png")


if __name__ == "__main__":
    main()
