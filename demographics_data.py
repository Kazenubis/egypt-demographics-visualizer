"""
Egypt population/demographics dataset — age structure and governorate
population growth, calibrated to real published statistics rather than
invented from scratch.

Sources (see README for the full list):
  - Broad-band age structure (male/female counts, ~2020): IndexMundi, citing
    CIA World Factbook-style estimates.
  - Governorate census figures (Cairo, Giza, Alexandria, multiple years):
    Wikipedia governorate pages, citing Egypt CAPMAS census data.

Both datasets are also exposed as tidy pandas DataFrames
(age_pyramid_dataframe, regional_growth_dataframe) for anyone who wants to
slice/aggregate them directly rather than through the helper functions
below.

The FINE-grained 5-year age pyramid bars are NOT independently sourced —
5-year-band-by-sex tables for Egypt are rendered as an interactive chart
(no accessible data table), not published as text. Instead, each real
broad band (0-14, 15-24, 25-54, 55-64, 65+) is split into standard 5-year
bins using an illustrative within-band distribution (front-loaded toward
the younger end of each band, tapering with age — matching the shape any
real developing-country pyramid has), scaled so the bins in each band sum
EXACTLY back to that band's real sourced total. This is the same
"synthetic but calibrated to real numbers" approach as the Cairo AQI
project from the previous batch, not independently-sourced fine data.
"""

import pandas as pd

# Real, sourced broad age-band population counts (~2020 estimate).
BROAD_AGE_BANDS = {
    "0-14": {"male": 18_112_550, "female": 16_889_155},
    "15-24": {"male": 9_684_437, "female": 9_071_163},
    "25-54": {"male": 20_032_310, "female": 19_376_847},
    "55-64": {"male": 3_160_438, "female": 3_172_544},
    "65+": {"male": 2_213_539, "female": 2_411_457},
}

# Illustrative within-band splits (NOT independently sourced — see module
# docstring). Each list of (bin_label, weight) must have weights summing to
# 1.0 for its parent band; verified in tests.
BAND_SPLITS = {
    "0-14": [("0-4", 0.360), ("5-9", 0.335), ("10-14", 0.305)],
    "15-24": [("15-19", 0.52), ("20-24", 0.48)],
    "25-54": [
        ("25-29", 0.20), ("30-34", 0.19), ("35-39", 0.18),
        ("40-44", 0.16), ("45-49", 0.14), ("50-54", 0.13),
    ],
    "55-64": [("55-59", 0.55), ("60-64", 0.45)],
    "65+": [("65-69", 0.42), ("70-74", 0.28), ("75-79", 0.18), ("80+", 0.12)],
}

# Real, sourced governorate census populations by year (irregular intervals
# — actual census/estimate years, not evenly spaced).
REGION_GROWTH = {
    "Cairo": [
        (1996, 6_800_991), (2006, 7_902_085), (2014, 9_102_232),
        (2018, 9_655_000), (2023, 10_203_693),
    ],
    "Giza": [
        (1996, 4_784_095), (2006, 6_294_319), (2019, 8_982_178), (2023, 9_456_137),
    ],
    "Alexandria": [
        (1986, 2_917_327), (1996, 3_339_076), (2006, 4_123_869),
        (2017, 5_163_750), (2023, 5_523_511),
    ],
}


def build_age_pyramid():
    """Returns a list of (age_bin, male_count, female_count) in ascending
    age order (youngest first), by splitting each real broad band into its
    5-year bins per BAND_SPLITS."""
    pyramid = []
    for band, split in BAND_SPLITS.items():
        band_totals = BROAD_AGE_BANDS[band]
        for bin_label, weight in split:
            male = round(band_totals["male"] * weight)
            female = round(band_totals["female"] * weight)
            pyramid.append((bin_label, male, female))
    return pyramid


def broad_band_totals_from_pyramid(pyramid):
    """Re-aggregates the fine-grained pyramid back into the original broad
    bands — used to verify the split didn't lose or invent population."""
    totals = {band: {"male": 0, "female": 0} for band in BROAD_AGE_BANDS}
    band_of_bin = {}
    for band, split in BAND_SPLITS.items():
        for bin_label, _ in split:
            band_of_bin[bin_label] = band

    for bin_label, male, female in pyramid:
        band = band_of_bin[bin_label]
        totals[band]["male"] += male
        totals[band]["female"] += female
    return totals


def age_pyramid_dataframe():
    """Tidy DataFrame version of build_age_pyramid(): one row per age bin,
    with male/female counts and each bin's share of total population."""
    df = pd.DataFrame(build_age_pyramid(), columns=["age_group", "male", "female"])
    df["total"] = df["male"] + df["female"]
    df["pct_of_population"] = df["total"] / df["total"].sum() * 100
    return df


def regional_growth_dataframe():
    """Tidy long-format DataFrame: one row per (region, year) census point."""
    rows = [
        {"region": region, "year": year, "population": pop}
        for region, points in REGION_GROWTH.items()
        for year, pop in points
    ]
    return pd.DataFrame(rows).sort_values(["region", "year"]).reset_index(drop=True)


def total_population():
    male = sum(b["male"] for b in BROAD_AGE_BANDS.values())
    female = sum(b["female"] for b in BROAD_AGE_BANDS.values())
    return male + female


def compute_cagr(start_pop, end_pop, start_year, end_year):
    """Compound annual growth rate, as a percentage."""
    years = end_year - start_year
    if years <= 0:
        raise ValueError("end_year must be after start_year")
    return ((end_pop / start_pop) ** (1 / years) - 1) * 100


def regional_growth_summary():
    """For each region, returns overall % growth and CAGR across its full
    recorded span, plus the most recent period's CAGR (the interesting
    "growth is slowing down" signal each of these governorates shows)."""
    summary = {}
    for region, points in REGION_GROWTH.items():
        points = sorted(points)
        first_year, first_pop = points[0]
        last_year, last_pop = points[-1]
        recent_start_year, recent_start_pop = points[-2]
        summary[region] = {
            "first_year": first_year,
            "last_year": last_year,
            "total_growth_pct": (last_pop / first_pop - 1) * 100,
            "overall_cagr": compute_cagr(first_pop, last_pop, first_year, last_year),
            "recent_cagr": compute_cagr(recent_start_pop, last_pop, recent_start_year, last_year),
        }
    return summary
