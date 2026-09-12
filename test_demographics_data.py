"""
Tests for demographics_data.py — the key check is that the fine-grained
age pyramid, split up for visual purposes, re-aggregates EXACTLY back to
the real sourced broad-band totals (no population invented or lost in the
split), plus sanity checks on the growth-rate math.
"""

import unittest

import demographics_data as dd


class TestBandSplitsSumToOne(unittest.TestCase):
    def test_every_bands_weights_sum_to_one(self):
        for band, split in dd.BAND_SPLITS.items():
            total_weight = sum(weight for _, weight in split)
            self.assertAlmostEqual(total_weight, 1.0, places=6, msg=f"band {band}")

    def test_every_broad_band_has_a_split(self):
        self.assertEqual(set(dd.BAND_SPLITS.keys()), set(dd.BROAD_AGE_BANDS.keys()))


class TestBuildAgePyramid(unittest.TestCase):
    def test_pyramid_has_one_row_per_bin(self):
        pyramid = dd.build_age_pyramid()
        expected_bins = sum(len(split) for split in dd.BAND_SPLITS.values())
        self.assertEqual(len(pyramid), expected_bins)

    def test_pyramid_is_in_ascending_age_order(self):
        pyramid = dd.build_age_pyramid()
        labels = [p[0] for p in pyramid]
        self.assertEqual(labels[0], "0-4")
        self.assertEqual(labels[-1], "80+")

    def test_reaggregating_the_pyramid_matches_the_real_sourced_totals(self):
        """The core calibration check: splitting each real broad band into
        5-year bins and summing them back up must reproduce the ORIGINAL
        sourced numbers exactly (within integer-rounding tolerance) — the
        fine split can't be allowed to drift from the real totals."""
        pyramid = dd.build_age_pyramid()
        reaggregated = dd.broad_band_totals_from_pyramid(pyramid)
        for band, real_totals in dd.BROAD_AGE_BANDS.items():
            self.assertAlmostEqual(
                reaggregated[band]["male"], real_totals["male"], delta=2,
                msg=f"{band} male total drifted from the sourced figure",
            )
            self.assertAlmostEqual(
                reaggregated[band]["female"], real_totals["female"], delta=2,
                msg=f"{band} female total drifted from the sourced figure",
            )


class TestTotalPopulation(unittest.TestCase):
    def test_total_population_is_plausible_for_egypt(self):
        # Sanity check against the real ~100-120M range these sourced
        # figures should fall in (not an exact figure, since sources are
        # from slightly different years).
        total = dd.total_population()
        self.assertGreater(total, 90_000_000)
        self.assertLess(total, 130_000_000)


class TestComputeCagr(unittest.TestCase):
    def test_known_doubling_over_ten_years_is_about_seven_percent(self):
        # A population that doubles in ~10 years grows at ~7.18%/yr (the
        # classic "rule of 70" ballpark, 70/10 ≈ 7).
        cagr = dd.compute_cagr(100, 200, 2000, 2010)
        self.assertAlmostEqual(cagr, 7.177, places=2)

    def test_zero_growth_gives_zero_cagr(self):
        self.assertAlmostEqual(dd.compute_cagr(100, 100, 2000, 2010), 0.0, places=6)

    def test_end_year_before_start_year_raises(self):
        with self.assertRaises(ValueError):
            dd.compute_cagr(100, 200, 2010, 2000)


class TestRegionalGrowthSummary(unittest.TestCase):
    def test_every_region_present_with_sensible_growth(self):
        summary = dd.regional_growth_summary()
        self.assertEqual(set(summary.keys()), set(dd.REGION_GROWTH.keys()))
        for region, s in summary.items():
            self.assertGreater(s["total_growth_pct"], 0, msg=f"{region} should show growth, not decline")
            self.assertLess(s["last_year"], 2030)
            self.assertGreater(s["last_year"], s["first_year"])

    def test_cairo_matches_hand_computed_total_growth(self):
        # 1996 -> 2023: 6,800,991 -> 10,203,693
        summary = dd.regional_growth_summary()
        expected_pct = (10_203_693 / 6_800_991 - 1) * 100
        self.assertAlmostEqual(summary["Cairo"]["total_growth_pct"], expected_pct, places=4)


class TestDataFrames(unittest.TestCase):
    def test_age_pyramid_dataframe_percentages_sum_to_100(self):
        df = dd.age_pyramid_dataframe()
        self.assertAlmostEqual(df["pct_of_population"].sum(), 100.0, places=3)
        self.assertEqual(set(df.columns), {"age_group", "male", "female", "total", "pct_of_population"})

    def test_regional_growth_dataframe_is_long_format_and_sorted(self):
        df = dd.regional_growth_dataframe()
        self.assertEqual(set(df.columns), {"region", "year", "population"})
        cairo_years = df[df["region"] == "Cairo"]["year"].tolist()
        self.assertEqual(cairo_years, sorted(cairo_years))
        self.assertEqual(len(df), sum(len(v) for v in dd.REGION_GROWTH.values()))


if __name__ == "__main__":
    unittest.main()
