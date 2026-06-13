import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
results_dir = root_dir / "data" / "robustness_results"

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

base_df = pd.read_json(results_dir / "base_robustness_results.jsonl", lines=True)
olmo_df = pd.read_json(results_dir / "olmo_base_robustness_results.jsonl", lines=True)

for df in (base_df, olmo_df):
    df["temperature"] = df["temperature"].round(1)

OCCUPATIONS = ["doctor", "nurse", "programmer", "scientist"]
PROMPT_STYLES = ["assumed", "author", "minimal"]
TEMPERATURES = [0.2, 0.5, 0.7, 1.0]

TEMP_MARKERS = {
    0.2: dict(marker="o", color="#B5D4F4"),
    0.5: dict(marker="^", color="#85B7EB"),
    0.7: dict(marker="s", color="#378ADD"),
    1.0: dict(marker="*", color="#185FA5"),
}

BAR_FACE = "#CECBF655"
BAR_EDGE = "#534AB7"


def gender_summary(df, group_cols):
    counts = df.groupby(group_cols + ["gender"]).size().unstack(fill_value=0)
    for col in ["female", "male", "none"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["total_mf"] = counts["female"] + counts["male"]
    counts["pct_female"] = 100 * counts["female"] / counts["total_mf"]
    counts["pct_none"] = 100 * counts["none"] / (counts["total_mf"] + counts["none"])
    return counts.reset_index()


OCC_INITIALS = "".join(o[0].upper() for o in OCCUPATIONS)
OC_INIT_HINT = "/".join(f"{o[0].upper()}={o}" for o in OCCUPATIONS)


def plot_gender_grid(df, model_order, model_labels, title, out_path):
    pooled = gender_summary(df, ["model_key", "prompt_style", "occupation"])
    per_temp = gender_summary(df, ["model_key", "prompt_style", "occupation", "temperature"])

    n_models = len(model_order)
    n_styles = len(PROMPT_STYLES)
    fig, axes = plt.subplots(
        n_models, n_styles, figsize=(4 * n_styles, 2.6 * n_models),
        sharex=True, sharey=True,
    )
    x = range(len(OCCUPATIONS))

    for i, model in enumerate(model_order):
        for j, style in enumerate(PROMPT_STYLES):
            ax = axes[i, j]

            sub_pooled = (
                pooled[(pooled.model_key == model) & (pooled.prompt_style == style)]
                .set_index("occupation").reindex(OCCUPATIONS)
            )
            ax.bar(x, sub_pooled["pct_female"], color=BAR_FACE, edgecolor=BAR_EDGE, width=0.6, zorder=2)

            for t in TEMPERATURES:
                sub_t = (
                    per_temp[(per_temp.model_key == model) & (per_temp.prompt_style == style) & (per_temp.temperature == t)]
                    .set_index("occupation").reindex(OCCUPATIONS)
                )
                m = TEMP_MARKERS[t]
                ax.scatter(x, sub_t["pct_female"], marker=m["marker"], color=m["color"],
                           edgecolor="#042C53", linewidth=0.5, s=35, zorder=3)

            # compact none-rate readout, e.g. "none% D0 N1 P0 S2"
            none_str = " ".join(f"{c}{p:.0f}" for c, p in zip(OCC_INITIALS, sub_pooled["pct_none"]))
            ax.text(0.02, 0.97, f"none% {none_str}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6, color="gray")

            ax.set_ylim(0, 105)
            ax.set_xticks(list(x))
            ax.set_xticklabels([o.capitalize() for o in OCCUPATIONS], rotation=0)
            ax.tick_params(axis="x", labelsize=8)

            if i == 0:
                ax.set_title(style.capitalize())
            if j == 0:
                ax.set_ylabel("% female")

    for i, model in enumerate(model_order):
        y = 1 - (i + 0.5) / n_models
        fig.text(0.005, y, model_labels[model], rotation=90, ha="left", va="center",
                 fontsize=10, fontweight="bold")

    legend_handles = [mpatches.Patch(facecolor=BAR_FACE, edgecolor=BAR_EDGE, label="pooled % female")]
    legend_handles += [
        mlines.Line2D([], [], color=m["color"], marker=m["marker"], linestyle="None",
                       markeredgecolor="#042C53", markersize=8, label=f"T={t}")
        for t, m in TEMP_MARKERS.items()
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=len(legend_handles), fontsize=9, frameon=False)

    fig.suptitle(title, y=1.08, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.01,
             f"'none% {OC_INIT_HINT}' = % of samples (pooled across temperatures) where gender could not be determined, per occupation",
             ha="center", va="top", fontsize=8, color="gray")
    fig.tight_layout(rect=(0.03, 0.01, 1, 1))
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


BASE_MODEL_ORDER = ["apertus-8b", "mistral-7b", "qwen2.5-7b", "llama-3.1-8b", "gemma-2-9b"]
BASE_MODEL_LABELS = {
    "apertus-8b": "Apertus",
    "mistral-7b": "Mistral",
    "qwen2.5-7b": "Qwen",
    "llama-3.1-8b": "Llama",
    "gemma-2-9b": "Gemma",
}

OLMO_MODEL_ORDER = ["stage1_final", "stage2_final", "stage3_final"]
OLMO_MODEL_LABELS = {
    "stage1_final": "Stage 1",
    "stage2_final": "Stage 2",
    "stage3_final": "Stage 3",
}

plot_gender_grid(base_df, BASE_MODEL_ORDER, BASE_MODEL_LABELS,
                  "Gender assignment by occupation and prompt style (base models)",
                  Path(__file__).parent / "_test_base.png")

plot_gender_grid(olmo_df, OLMO_MODEL_ORDER, OLMO_MODEL_LABELS,
                  "Gender assignment by occupation and prompt style (Olmo training stages)",
                  Path(__file__).parent / "_test_olmo.png")

print("done")
