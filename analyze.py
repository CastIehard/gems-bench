"""Visualize GEMS-Bench results per mode. Run after judge.py.

Writes plots to results/plots/ and a summary table (markdown + CSV), including
a per-category accuracy heatmap and a mean set-F1 column for list answers.
Requires results/results.json.

Usage:
    python analyze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import orjson
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import load_config  # noqa: E402

_CFG = load_config()
_PATHS = _CFG["_paths"]
_ANALYZE = _CFG["analyze"]
PLOTS_DIR = _PATHS["plots_dir"]
RESULTS_PATH = _PATHS["results"]

# One hue, light to dark, for magnitude. Accuracy has no meaningful midpoint, so a
# diverging red-to-green scale would invent one — and red/green is the pair that
# collapses for the most common colour-vision deficiency.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def load_frame() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found — run judge.py first")
    results = orjson.loads(RESULTS_PATH.read_bytes())
    rows = []
    for mode, mode_results in results.items():
        for question_id, record in mode_results.items():
            rows.append(
                {
                    "mode": mode,
                    "question_id": question_id,
                    "category": record.get("category"),
                    # None for synthetic audio; set for human recordings so the
                    # 50/50 speaker split can be checked for a voice effect
                    "speaker": record.get("speaker"),
                    "audio_source": record.get("audio_source"),
                    "status": record.get("status"),
                    "correct": record.get("decision") == "correct",
                    "list_f1": (record.get("list_f1") or {}).get("f1"),
                    "ttft_s": record.get("ttft_s"),
                    "ttfa_s": record.get("ttfa_s"),
                    "tools_during": record.get("tools_during_listening", 0),
                    "tools_after": record.get("tools_after_listening", 0),
                    # retrieval quality over prose docs (judge.py extraction)
                    "doc_precision": record.get("doc_precision"),
                    "doc_recall": record.get("doc_recall"),
                    "gold_doc_recall_eos": record.get("gold_doc_recall_eos"),
                    "answer_word_count": record.get("answer_word_count"),
                }
            )
    if not rows:
        raise SystemExit("results.json contains no judged runs")
    frame = pd.DataFrame(rows)
    # Plot/table order = the order the modes appear in results.json, so the
    # benchmark never needs to know your mode labels in advance.
    order = list(frame["mode"].unique())
    frame["mode"] = pd.Categorical(frame["mode"], categories=order, ordered=True)
    return frame.sort_values(["mode", "question_id"])


def plot_accuracy(frame: pd.DataFrame, plt) -> None:
    accuracy = frame.groupby("mode", observed=True)["correct"].mean()
    ax = accuracy.plot.bar(rot=0, color="#4c72b0", figsize=(6, 4))
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("GEMS accuracy per mode (deterministic)")
    for index, value in enumerate(accuracy):
        ax.text(index, value + 0.02, f"{value:.0%}", ha="center")
    ax.figure.tight_layout()
    ax.figure.savefig(PLOTS_DIR / "accuracy_per_mode.png", dpi=150)
    plt.close(ax.figure)


def plot_latency(frame: pd.DataFrame, plt) -> None:
    for metric, label in (
        ("ttfa_s", "Time to first audio (s)"),
        ("ttft_s", "Time to first token (s)"),
    ):
        data = frame.dropna(subset=[metric])
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        modes = [m for m in data["mode"].cat.categories if m in set(data["mode"])]
        ax.boxplot(
            [data.loc[data["mode"] == mode, metric] for mode in modes],
            tick_labels=modes,
        )
        for index, mode in enumerate(modes, start=1):
            values = data.loc[data["mode"] == mode, metric]
            ax.plot([index] * len(values), values, "o", alpha=0.5, color="#dd8452")
        ax.set_ylabel(label)
        ax.set_title(f"{label} per mode")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{metric.rstrip('_s')}_per_mode.png", dpi=150)
        plt.close(fig)


def plot_accuracy_vs_ttfa(frame: pd.DataFrame, plt) -> None:
    """One point per mode: median TTFA (x) against accuracy (y).

    The trade-off view — bottom-right is slow-and-wrong, top-left is the goal
    (fast and accurate). Identity via direct labels, not color."""
    stats = frame.groupby("mode", observed=True).agg(
        accuracy=("correct", "mean"), ttfa_median=("ttfa_s", "median")
    )
    stats = stats.dropna()
    if stats.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(stats["ttfa_median"], stats["accuracy"], s=90, color="#4c72b0", zorder=3)
    for mode, row in stats.iterrows():
        ax.annotate(
            f"{mode}\n({row['accuracy']:.0%}, {row['ttfa_median']:.1f}s)",
            (row["ttfa_median"], row["accuracy"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )
    ax.set_xlabel("Time to first audio, median (s)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, stats["ttfa_median"].max() * 1.6)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_title("Accuracy vs. response latency per mode")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_vs_ttfa.png", dpi=150)
    plt.close(fig)


def plot_tool_calls(frame: pd.DataFrame, plt) -> None:
    grouped = frame.groupby("mode", observed=True)[
        ["tools_during", "tools_after"]
    ].sum()
    ax = grouped.plot.bar(
        stacked=True, rot=0, figsize=(6, 4), color=["#55a868", "#c44e52"]
    )
    ax.set_ylabel("Tool calls (total)")
    ax.set_title("Tool calls during vs after listening")
    ax.legend(["during listening", "after listening"])
    ax.figure.tight_layout()
    ax.figure.savefig(PLOTS_DIR / "tool_calls_split.png", dpi=150)
    plt.close(ax.figure)


def plot_question_heatmap(frame: pd.DataFrame, plt) -> None:
    pivot = frame.pivot_table(
        index="question_id",
        columns="mode",
        values="correct",
        aggfunc="first",
        observed=True,
    ).astype(float)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(pivot) + 2))
    image = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Correct (green) / incorrect (red) per question × mode")
    fig.colorbar(image, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "question_mode_heatmap.png", dpi=150)
    plt.close(fig)


def plot_category_heatmap(frame: pd.DataFrame, plt) -> None:
    """Accuracy per category × mode.

    Accuracy is a magnitude, so the cells use ONE hue from light to dark rather
    than a red-to-green scale: a diverging scale implies a neutral midpoint that
    does not exist here, and red/green is the one pair that collapses for the
    most common colour-vision deficiency. Rows follow the order the categories
    are declared in config.yaml, which runs from the control case to the hardest.
    """
    from matplotlib.colors import LinearSegmentedColormap

    order = [c for c in _CFG["questions"]["distribution"] if c in set(frame["category"])]
    pivot = (
        frame.pivot_table(
            index="category",
            columns="mode",
            values="correct",
            aggfunc="mean",
            observed=True,
        )
        .astype(float)
        .reindex(order)
    )
    ramp = LinearSegmentedColormap.from_list("gems_blue", SEQUENTIAL_BLUE)

    fig, ax = plt.subplots(figsize=(1.9 * len(pivot.columns) + 2.6, 0.62 * len(pivot) + 2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(pivot.values, cmap=ramp, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.tick_params(length=0, colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 2 px of surface between cells, so neighbouring blues stay separable
    ax.set_xticks([x - 0.5 for x in range(1, len(pivot.columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(pivot.index))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if pd.notna(val):
                # ink flips on the dark half of the ramp, where dark text dies
                ax.text(
                    j, i, f"{val:.0%}",
                    ha="center", va="center", fontsize=10,
                    color=SURFACE if val > 0.55 else TEXT_PRIMARY,
                )
    ax.set_title("Accuracy per category × mode", color=TEXT_PRIMARY, pad=12)
    bar = fig.colorbar(image, ax=ax, shrink=0.7)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "category_mode_heatmap.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_concealment(frame: pd.DataFrame, plt) -> None:
    """Share of retrieval done inside the listening window per mode:
    during / (during + after). Higher = less retrieval left for after the user
    stops speaking."""
    grouped = frame.groupby("mode", observed=True)[
        ["tools_during", "tools_after"]
    ].sum()
    total = grouped["tools_during"] + grouped["tools_after"]
    ratio = (grouped["tools_during"] / total.where(total > 0)).fillna(0.0)
    ax = ratio.plot.bar(rot=0, color="#8172b3", figsize=(6, 4))
    ax.set_ylabel("Concealment ratio (during / total tool calls)")
    ax.set_ylim(0, 1)
    ax.set_title("Share of retrieval done inside the listening window")
    for index, value in enumerate(ratio):
        ax.text(index, value + 0.02, f"{value:.0%}", ha="center")
    ax.figure.tight_layout()
    ax.figure.savefig(PLOTS_DIR / "concealment_ratio.png", dpi=150)
    plt.close(ax.figure)


def plot_doc_precision(frame: pd.DataFrame, plt) -> None:
    """Retrieval precision per mode: of the docs the search tool returned, the
    share that were actually required (gold). "90 % required, 10 % not" = 0.90.
    Also overlays recall (gold docs found / gold docs)."""
    data = frame.dropna(subset=["doc_precision"])
    if data.empty:
        print("  doc precision: no doc_precision in results — re-run judge.py")
        return
    grouped = data.groupby("mode", observed=True)[
        ["doc_precision", "doc_recall"]
    ].mean()
    ax = grouped.plot.bar(rot=0, figsize=(6, 4), color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("Mean over questions")
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval precision (required / retrieved) & recall per mode")
    ax.legend(["precision (required ÷ retrieved)", "recall (gold found ÷ gold)"])
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.0f%%",
            label_type="edge",
            padding=2,
            labels=[f"{v:.0%}" for v in container.datavalues],
        )
    ax.figure.tight_layout()
    ax.figure.savefig(PLOTS_DIR / "doc_precision.png", dpi=150)
    plt.close(ax.figure)


def report_deadline(frame: pd.DataFrame) -> None:
    """Gold-Doc-Recall @ EOS per mode: the fraction of gold evidence retrieved
    before the user stops speaking. judge.py extracts it from
    raw_realtime_events.jsonl (returned doc-ids) joined against the EOS
    timestamp in timing.jsonl and emits `gold_doc_recall_eos`. Reports the
    per-mode mean; a results.json predating that extraction shows unavailable
    rather than a fabricated number."""
    ns = _ANALYZE["deadline_roundtrips"]
    if frame["gold_doc_recall_eos"].notna().any():
        by_mode = (
            frame.dropna(subset=["gold_doc_recall_eos"])
            .groupby("mode", observed=True)["gold_doc_recall_eos"]
            .mean()
            .round(3)
        )
        print(
            "  deadline — Gold-Doc-Recall @ EOS (fraction of gold evidence "
            "gathered before end-of-speech):"
        )
        for mode, val in by_mode.items():
            print(f"    {mode}: {val:.0%}")
        return
    print(
        "  deadline: results.json carries no gold_doc_recall_eos — re-run "
        "judge.py to extract it from raw_realtime_events.jsonl. Reporting "
        f"nothing rather than a substitute number. Declared N = {ns}."
    )


def report_speaker(frame: pd.DataFrame) -> None:
    """Speaker robustness check, human channel only.

    The 50/50 split is stratified per category (recording.speakers), so each
    speaker covers every category equally and a gap between the two is a VOICE
    effect rather than a category effect.

    Restricted to rows whose `audio_source` is a human channel. Every item
    carries a speaker in the manifest — that is who WOULD read it — so on
    synthesized audio the same two groups exist but share one TTS voice. A gap
    there is item difficulty between two halves of the set, not a voice effect,
    and printing it under this heading would invite exactly the wrong reading.
    """
    human = frame[frame["audio_source"] == "real"]
    if human.empty or not human["speaker"].notna().any():
        if frame["audio_source"].isna().any():
            print(
                "\n  speaker robustness: skipped — results carry no audio_source, "
                "so the channel is unknown (re-run the driver, or backfill it)"
            )
        return
    by_speaker = (
        human.dropna(subset=["speaker"])
        .groupby(["mode", "speaker"], observed=True)
        .agg(
            questions=("question_id", "count"),
            accuracy=("correct", "mean"),
            ttfa_median_s=("ttfa_s", "median"),
        )
        .round(3)
    )
    print("\n  speaker robustness (stratified 50/50 — a gap here is a voice effect):")
    print(by_speaker.to_markdown())


def write_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("mode", observed=True)
        .agg(
            questions=("question_id", "count"),
            accuracy=("correct", "mean"),
            list_f1_mean=("list_f1", "mean"),
            ttfa_mean_s=("ttfa_s", "mean"),
            ttfa_median_s=("ttfa_s", "median"),
            ttft_mean_s=("ttft_s", "mean"),
            tools_during=("tools_during", "sum"),
            tools_after=("tools_after", "sum"),
            # critical-path round-trips: retrieval rounds AFTER the user stops
            # (speed-independent). Mean per question.
            critical_path_roundtrips=("tools_after", "mean"),
            # retrieval quality: required ÷ retrieved, and gold found ÷ gold
            doc_precision_mean=("doc_precision", "mean"),
            doc_recall_mean=("doc_recall", "mean"),
            gold_doc_recall_eos_mean=("gold_doc_recall_eos", "mean"),
            # verbosity covariate: if this tracks accuracy across modes, the
            # scorer's whole-transcript matching may be inflating the verbose
            # mode — report it so the (non-)correlation is visible, not assumed.
            answer_words_mean=("answer_word_count", "mean"),
        )
        .round(3)
    )
    total = summary["tools_during"] + summary["tools_after"]
    summary["concealment_ratio"] = (
        (summary["tools_during"] / total.where(total > 0)).fillna(0.0).round(3)
    )
    summary.to_csv(PLOTS_DIR / "summary.csv")
    (PLOTS_DIR / "summary.md").write_text(summary.to_markdown())
    return summary


def run(show: bool = False) -> pd.DataFrame:
    """Generate plots + summary from results.json. Returns summary DataFrame."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = load_frame()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_accuracy(frame, plt)
    plot_latency(frame, plt)
    plot_accuracy_vs_ttfa(frame, plt)
    plot_tool_calls(frame, plt)
    if _ANALYZE["concealment"]:
        plot_concealment(frame, plt)
    plot_doc_precision(frame, plt)
    plot_question_heatmap(frame, plt)
    plot_category_heatmap(frame, plt)
    summary = write_summary(frame)

    print(summary.to_markdown())
    report_deadline(frame)
    report_speaker(frame)
    print(f"\nPlots + summary written to {PLOTS_DIR}")
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
