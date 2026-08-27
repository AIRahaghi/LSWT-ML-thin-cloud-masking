from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import pandas as pd

from lswt_cloud_masking.leave_one_site_out import (
    LeaveOneSiteOutConfig,
    _run_fingerprint,
    _scene_partition_manifest,
    _scene_season,
    _summarize_results,
    create_leave_one_site_split,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SITES = {"ageri", "bianco", "geneva", "greifensee", "mendota", "venice"}
OVERLAP_EXCEPTION = "LC08_L1TP_196028_20210906_20210915_02_T1"


class LeaveOneSiteOutSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = ROOT / "configs" / "leave_one_site_out.example.json"
        cls.config = LeaveOneSiteOutConfig.from_json(config_path)
        cls.source = pd.concat(
            [pd.read_csv(ROOT / path) for path in cls.config.input_csvs],
            ignore_index=True,
        )

    def test_example_config_uses_six_sites_and_consistent_trial_budgets(self) -> None:
        self.assertEqual(set(self.config.held_out_sites or []), EXPECTED_SITES)
        self.assertEqual(self.config.cv_splits, 5)
        self.assertEqual(self.config.scoring, "balanced_accuracy")
        self.assertEqual(self.config.overlap_exception_scene_id, OVERLAP_EXCEPTION)
        self.assertEqual(self.config.overlap_exception_site, "geneva")
        self.assertEqual(self.config.overlap_exception_class, 3)
        self.assertEqual(
            (
                self.config.overlap_exception_train_rows,
                self.config.overlap_exception_test_rows,
            ),
            (120, 30),
        )
        trial_budgets = (
            self.config.n_trials_dt,
            self.config.n_trials_rf,
            self.config.n_trials_xgb,
        )
        self.assertTrue(all(trials > 0 for trials in trial_budgets))
        self.assertEqual(len(set(trial_budgets)), 1)

    def test_every_real_site_split_is_complete_and_exclusive(self) -> None:
        total_rows = len(self.source)
        all_sensors = set(self.source["scene_id"].str[:4])
        for held_out_site in sorted(EXPECTED_SITES):
            with self.subTest(held_out_site=held_out_site):
                splits, summary = create_leave_one_site_split(
                    self.source,
                    held_out_site,
                    self.config,
                )
                held_out_rows = int(self.source["lakeN"].eq(held_out_site).sum())
                development_rows = total_rows - held_out_rows

                self.assertEqual(len(splits["leave_out_site"]), held_out_rows)
                self.assertAlmostEqual(
                    len(splits["test"]) / development_rows,
                    0.2,
                    delta=0.002,
                )
                self.assertEqual(
                    sum(len(frame) for frame in splits.values()),
                    total_rows,
                )
                self.assertEqual(
                    set(splits["leave_out_site"]["lakeN"]),
                    {held_out_site},
                )
                for name in ("train", "test"):
                    self.assertNotIn(held_out_site, set(splits[name]["lakeN"]))
                    self.assertEqual(set(splits[name]["lakeN"]), EXPECTED_SITES - {held_out_site})
                for frame in splits.values():
                    self.assertEqual(set(frame["lst_class"].astype(int)), {1, 2, 3})
                    self.assertEqual(set(frame["scene_id"].str[:4]), all_sensors)
                for name in ("train", "test"):
                    self.assertEqual(
                        set(_scene_season(splits[name]["scene_id"])),
                        {"DJF", "MAM", "JJA", "SON"},
                    )

                train_test_overlap = set(splits["train"]["scene_id"]) & set(
                    splits["test"]["scene_id"]
                )
                expected_overlap = (
                    set() if held_out_site == "geneva" else {OVERLAP_EXCEPTION}
                )
                self.assertEqual(train_test_overlap, expected_overlap)
                exception_counts = {
                    name: int(frame["scene_id"].eq(OVERLAP_EXCEPTION).sum())
                    for name, frame in splits.items()
                }
                self.assertEqual(
                    exception_counts,
                    {"train": 0, "test": 0, "leave_out_site": 150}
                    if held_out_site == "geneva"
                    else {"train": 120, "test": 30, "leave_out_site": 0},
                )
                manifest = _scene_partition_manifest(splits, self.config)
                manifest_overlap = set(
                    manifest.loc[manifest["is_train_test_overlap"], "scene_id"]
                )
                self.assertEqual(manifest_overlap, expected_overlap)
                self.assertTrue(summary["validation"]["held_out_site_exclusive"])
                self.assertTrue(
                    summary["validation"]["only_configured_train_test_overlap"]
                )
                expected_shared_scenes = set(
                    self.source.loc[
                        self.source["lakeN"].eq(held_out_site), "scene_id"
                    ]
                ) & set(
                    self.source.loc[
                        ~self.source["lakeN"].eq(held_out_site), "scene_id"
                    ]
                )
                self.assertEqual(
                    summary["validation"]["shared_acquisition_scene_count"],
                    len(expected_shared_scenes),
                )

    def test_split_is_deterministic(self) -> None:
        first, _ = create_leave_one_site_split(self.source, "geneva", self.config)
        second, _ = create_leave_one_site_split(self.source, "geneva", self.config)
        for name in first:
            self.assertListEqual(
                first[name]["_source_uid"].tolist(),
                second[name]["_source_uid"].tolist(),
            )

    def test_resume_fingerprint_ignores_site_subset_and_resume_flag(self) -> None:
        input_records = [{"path": "source.csv", "rows": 10, "sha256": "abc"}]
        first = _run_fingerprint(self.config, "geneva", input_records)
        orchestration_change = replace(
            self.config,
            held_out_sites=["geneva"],
            resume_completed_runs=not self.config.resume_completed_runs,
        )
        self.assertEqual(
            first,
            _run_fingerprint(orchestration_change, "geneva", input_records),
        )
        scientific_change = replace(self.config, n_trials_rf=121)
        self.assertNotEqual(
            first,
            _run_fingerprint(scientific_change, "geneva", input_records),
        )


class LeaveOneSiteOutMetricTests(unittest.TestCase):
    def test_summary_reports_site_robustness_distribution(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "model": "random_forest",
                    "train_balanced_accuracy": 0.95,
                    "test_balanced_accuracy": 0.85,
                    "leave_out_site_balanced_accuracy": 0.55,
                },
                {
                    "model": "random_forest",
                    "train_balanced_accuracy": 0.97,
                    "test_balanced_accuracy": 0.87,
                    "leave_out_site_balanced_accuracy": 0.65,
                },
            ]
        )
        summary = _summarize_results(results).iloc[0]
        self.assertEqual(summary["n_sites"], 2)
        self.assertAlmostEqual(summary["test_balanced_accuracy_mean"], 0.86)
        self.assertAlmostEqual(summary["leave_out_site_balanced_accuracy_mean"], 0.60)
        self.assertGreater(summary["leave_out_site_balanced_accuracy_std"], 0)


if __name__ == "__main__":
    unittest.main()
