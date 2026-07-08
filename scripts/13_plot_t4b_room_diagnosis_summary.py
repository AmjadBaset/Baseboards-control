from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path(
    "data/processed/twin4build/two_room_restricted_flow/"
    "t4b_room_level_diagnosis_summary.csv"
)

OUT_DIR = Path(
    "reports/figures/twin4build/two_room_restricted_flow"
)


def save_bar_plot(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    output_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))

    x = df["zone"].astype(str)
    y = df[y_col]

    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel("Room")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, y.max() * 1.15))
    ax.tick_params(axis="x", rotation=35)

    for i, value in enumerate(y):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = OUT_DIR / output_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


def save_valve_reference_plot(df: pd.DataFrame) -> None:
    required = ["zone", "valve_position_mean", "normalized_flow_fraction_mean_P90"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Skipping valve reference plot. Missing columns: {missing}")
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))

    x = range(len(df))
    width = 0.38

    ax.bar(
        [i - width / 2 for i in x],
        df["valve_position_mean"],
        width=width,
        label="T4B valve position mean",
    )
    ax.bar(
        [i + width / 2 for i in x],
        df["normalized_flow_fraction_mean_P90"],
        width=width,
        label="OpenStudio reference P90",
    )

    ax.set_title("Valve effort compared with OpenStudio reference")
    ax.set_xlabel("Room")
    ax.set_ylabel("Valve position / normalized flow fraction")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["zone"], rotation=35)
    ax.set_ylim(
        0,
        max(
            1.0,
            df["valve_position_mean"].max() * 1.15,
            df["normalized_flow_fraction_mean_P90"].max() * 1.15,
        ),
    )
    ax.legend()

    fig.tight_layout()
    out_path = OUT_DIR / "t4b_valve_effort_vs_reference_p90.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_PATH)
    df = df.sort_values("zone").copy()

    save_bar_plot(
        df,
        y_col="dominant_fault_fraction",
        title="Dominant fault fraction by room",
        ylabel="Fraction of diagnosed windows",
        output_name="t4b_dominant_fault_fraction_by_room.png",
    )

    save_bar_plot(
        df,
        y_col="normal_fraction",
        title="Normal window fraction by room",
        ylabel="Fraction of normal windows",
        output_name="t4b_normal_fraction_by_room.png",
    )

    save_valve_reference_plot(df)


if __name__ == "__main__":
    main()
