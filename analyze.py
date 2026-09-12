"""CLI + chart rendering for the Egypt demographics dataset."""

import argparse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")

import demographics_data as dd

MALE_COLOR = "#2f6690"
FEMALE_COLOR = "#c9184a"


def plot_age_pyramid(pyramid_df, output_path):
    labels = pyramid_df["age_group"].tolist()
    male = (-pyramid_df["male"]).tolist()
    female = pyramid_df["female"].tolist()
    y_pos = range(len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y_pos, male, color=MALE_COLOR, label="Male")
    ax.barh(y_pos, female, color=FEMALE_COLOR, label="Female")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x) / 1e6:.1f}M"))
    ax.set_xlabel("Population")
    ax.set_ylabel("Age group")
    ax.set_title("Egypt Population Pyramid\n(calibrated to real ~2020 broad-band age structure)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_regional_growth(growth_df, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for region, group in growth_df.groupby("region"):
        group = group.sort_values("year")
        ax.plot(
            group["year"], group["population"] / 1e6,
            marker="o", linewidth=2, label=region,
        )

    ax.set_xlabel("Census / estimate year")
    ax.set_ylabel("Population (millions)")
    ax.set_title("Governorate Population Growth\n(real census figures — Cairo, Giza, Alexandria)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Egypt demographics visualizer")
    parser.add_argument("--pyramid-out", default="assets/age_pyramid.png")
    parser.add_argument("--growth-out", default="assets/regional_growth.png")
    args = parser.parse_args()

    pyramid_df = dd.age_pyramid_dataframe()
    growth_df = dd.regional_growth_dataframe()
    plot_age_pyramid(pyramid_df, args.pyramid_out)
    plot_regional_growth(growth_df, args.growth_out)

    print(f"Total (sourced broad-band) population: {dd.total_population():,}")
    print()
    print("Age pyramid (fine bins, calibrated to real broad-band totals):")
    print(pyramid_df.to_string(index=False, formatters={
        "male": "{:,}".format, "female": "{:,}".format, "total": "{:,}".format,
        "pct_of_population": "{:.2f}%".format,
    }))
    print()
    print("Regional growth summary:")
    for region, s in dd.regional_growth_summary().items():
        print(
            f"  {region}: {s['first_year']}->{s['last_year']} "
            f"+{s['total_growth_pct']:.1f}% total "
            f"({s['overall_cagr']:.2f}%/yr overall, {s['recent_cagr']:.2f}%/yr most recent period)"
        )
    print()
    print(f"Saved: {args.pyramid_out}, {args.growth_out}")


if __name__ == "__main__":
    main()
