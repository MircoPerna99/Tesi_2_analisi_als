import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# ============================================================
# Configurazione
# ============================================================

OUTPUT_DIR = Path("grafici")
OUTPUT_DIR.mkdir(exist_ok=True)

TOP_PERCENTAGES = [1, 5, 10, 30, 50]


# ============================================================
# Dati
# ============================================================

data = [
    # ALS - GS1 - Reliability Yes
    ["ALS", "GS1", "Yes", 2588, 1, 0.083, 0.688],
    ["ALS", "GS1", "Yes", 2588, 5, 0.291, 1.692],
    ["ALS", "GS1", "Yes", 2588, 10, 0.489, 2.396],
    ["ALS", "GS1", "Yes", 2588, 30, 0.679, 1.536],
    ["ALS", "GS1", "Yes", 2588, 50, 0.817, 1.334],

    # ALS - GS1 - Reliability No
    ["ALS", "GS1", "No", 2588, 1, 0.143, 2.055],
    ["ALS", "GS1", "No", 2588, 5, 0.245, 1.200],
    ["ALS", "GS1", "No", 2588, 10, 0.300, 0.901],
    ["ALS", "GS1", "No", 2588, 30, 0.406, 0.549],
    ["ALS", "GS1", "No", 2588, 50, 0.481, 0.463],

    # ALS - GS2 - Reliability Yes
    ["ALS", "GS2", "Yes", 172, 1, 0.186, 3.461],
    ["ALS", "GS2", "Yes", 172, 5, 0.355, 2.516],
    ["ALS", "GS2", "Yes", 172, 10, 0.465, 2.163],
    ["ALS", "GS2", "Yes", 172, 30, 0.959, 3.068],
    ["ALS", "GS2", "Yes", 172, 50, 0.994, 1.977],

    # ALS - GS2 - Reliability No
    ["ALS", "GS2", "No", 172, 1, 0.267, 7.152],
    ["ALS", "GS2", "No", 172, 5, 0.500, 5.000],
    ["ALS", "GS2", "No", 172, 10, 0.616, 3.798],
    ["ALS", "GS2", "No", 172, 30, 0.901, 2.707],
    ["ALS", "GS2", "No", 172, 50, 0.965, 1.863],

    # ALS - GS3 - Reliability Yes
    ["ALS", "GS3", "Yes", 200, 1, 0.196, 3.827],
    ["ALS", "GS3", "Yes", 200, 5, 0.332, 2.198],
    ["ALS", "GS3", "Yes", 200, 10, 0.473, 2.236],
    ["ALS", "GS3", "Yes", 200, 30, 0.946, 2.981],
    ["ALS", "GS3", "Yes", 200, 50, 0.984, 1.935],

    # ALS - GS3 - Reliability No
    ["ALS", "GS3", "No", 200, 1, 0.284, 8.036],
    ["ALS", "GS3", "No", 200, 5, 0.500, 5.000],
    ["ALS", "GS3", "No", 200, 10, 0.655, 4.285],
    ["ALS", "GS3", "No", 200, 30, 0.907, 2.743],
    ["ALS", "GS3", "No", 200, 50, 0.969, 1.878],

    # WGNN-DTA - GS3 - Reliability Yes
    ["WGNN-DTA", "GS3", "Yes", 200, 1, 0.255, 6.523],
    ["WGNN-DTA", "GS3", "Yes", 200, 5, 0.663, 8.792],
    ["WGNN-DTA", "GS3", "Yes", 200, 10, 0.859, 7.373],
    ["WGNN-DTA", "GS3", "Yes", 200, 30, 1.000, 3.333],
    ["WGNN-DTA", "GS3", "Yes", 200, 50, 1.000, 2.000],

    # ALS+WGNN-DTA - GS3 - Reliability Yes
    ["ALS+WGNN-DTA", "GS3", "Yes", 200, 1, 0.190, 12.021],
    ["ALS+WGNN-DTA", "GS3", "Yes", 200, 5, 0.321, 7.770],
    ["ALS+WGNN-DTA", "GS3", "Yes", 200, 10, 0.451, 8.266],
    ["ALS+WGNN-DTA", "GS3", "Yes", 200, 30, 0.886, 13.041],
    ["ALS+WGNN-DTA", "GS3", "Yes", 200, 50, 0.913, 9.316],
]

columns = [
    "Method",
    "Gold Standard",
    "Reliability",
    "Size",
    "Top Percentage",
    "Recall",
    "Lee-Liu score",
]

df = pd.DataFrame(data, columns=columns)


# ============================================================
# Funzione comune per la formattazione
# ============================================================

def format_plot(
    ax,
    title,
    ylabel,
    y_min=None,
    y_max=None,
):
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Top ranking (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticks(TOP_PERCENTAGES)
    ax.set_xticklabels([f"{value}%" for value in TOP_PERCENTAGES])

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=None, frameon=True)
    ax.figure.tight_layout()


# ============================================================
# 1-2. Grafici ALS per Gold Standard e reliability threshold
# ============================================================

def plot_als_by_gold_standard(metric, ylabel, output_filename):
    als_data = df[df["Method"] == "ALS"].copy()

    line_styles = {
        "Yes": "-",
        "No": "--",
    }

    markers = {
        "GS1": "o",
        "GS2": "s",
        "GS3": "^",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for gold_standard in ["GS1", "GS2", "GS3"]:
        for reliability in ["Yes", "No"]:
            subset = als_data[
                (als_data["Gold Standard"] == gold_standard)
                & (als_data["Reliability"] == reliability)
            ].sort_values("Top Percentage")

            threshold_label = (
                "with threshold"
                if reliability == "Yes"
                else "without threshold"
            )

            ax.plot(
                subset["Top Percentage"],
                subset[metric],
                marker=markers[gold_standard],
                linestyle=line_styles[reliability],
                linewidth=2,
                markersize=7,
                label=f"{gold_standard} - {threshold_label}",
            )

    y_max = 1.05 if metric == "Recall" else None

    format_plot(
        ax=ax,
        title=f"ALS: {ylabel} per Gold Standard",
        ylabel=ylabel,
        y_min=0,
        y_max=y_max,
    )

    fig.savefig(
        OUTPUT_DIR / output_filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


plot_als_by_gold_standard(
    metric="Recall",
    ylabel="Recall",
    output_filename="als_recall_gold_standard_threshold.png",
)

plot_als_by_gold_standard(
    metric="Lee-Liu score",
    ylabel="Lee–Liu score",
    output_filename="als_lee_liu_gold_standard_threshold.png",
)


# ============================================================
# 3-4. Confronto dei metodi su GS3
# ============================================================

def plot_gs3_method_comparison(metric, ylabel, output_filename):
    methods = ["ALS", "WGNN-DTA", "ALS+WGNN-DTA"]

    gs3_data = df[
        (df["Gold Standard"] == "GS3")
        & (df["Reliability"] == "Yes")
        & (df["Method"].isin(methods))
    ].copy()

    markers = {
        "ALS": "o",
        "WGNN-DTA": "s",
        "ALS+WGNN-DTA": "^",
    }

    line_styles = {
        "ALS": "-",
        "WGNN-DTA": "--",
        "ALS+WGNN-DTA": "-.",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        subset = (
            gs3_data[gs3_data["Method"] == method]
            .sort_values("Top Percentage")
        )

        if subset.empty:
            print(f"Attenzione: nessun dato trovato per {method}")
            continue

        x_values = subset["Top Percentage"].astype(float).to_numpy()
        y_values = subset[metric].astype(float).to_numpy()

        ax.plot(
            x_values,
            y_values,
            marker=markers[method],
            linestyle=line_styles[method],
            linewidth=2,
            markersize=7,
            label=method,
        )

    ax.set_title(
        f"Confronto dei metodi su GS3: {ylabel}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Top ranking (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticks(TOP_PERCENTAGES)
    ax.set_xticklabels(
        [f"{percentage}%" for percentage in TOP_PERCENTAGES]
    )

    ax.set_ylim(bottom=0)

    if metric == "Recall":
        ax.set_ylim(0, 1.05)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_DIR / output_filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Grafico salvato in: {output_path}")

    # Con il backend Agg questa operazione non usa finestre macOS.
    plt.close(fig)

plot_gs3_method_comparison(
    metric="Recall",
    ylabel="Recall",
    output_filename="gs3_metodi_recall.png",
)

plot_gs3_method_comparison(
    metric="Lee-Liu score",
    ylabel="Lee–Liu score",
    output_filename="gs3_metodi_lee_liu.png",
)