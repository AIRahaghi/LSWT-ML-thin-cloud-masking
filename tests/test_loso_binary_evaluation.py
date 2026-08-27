from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lswt_cloud_masking.loso_binary_evaluation import (
    _binary_comparison_table,
    _summarize_binary_results,
    binary_cloud_water_metrics,
)


class BinaryCloudWaterMetricTests(unittest.TestCase):
    def test_binary_metrics_merge_original_cloud_classes(self) -> None:
        metrics = binary_cloud_water_metrics(
            np.array([1, 2, 3, 3]),
            np.array([1, 3, 2, 3]),
        )
        self.assertEqual(metrics["actual_total_cloud_rows"], 2)
        self.assertEqual(metrics["actual_water_rows"], 2)
        self.assertEqual(metrics["true_total_cloud_predicted_total_cloud"], 1)
        self.assertEqual(metrics["true_total_cloud_predicted_water"], 1)
        self.assertEqual(metrics["true_water_predicted_total_cloud"], 1)
        self.assertEqual(metrics["true_water_predicted_water"], 1)
        self.assertAlmostEqual(metrics["overall_binary_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_binary_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["total_cloud_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["water_accuracy"], 0.5)

    def test_binary_metrics_reject_unknown_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected predicted labels"):
            binary_cloud_water_metrics(
                np.array([1, 2, 3]),
                np.array([1, 2, 4]),
            )

    def test_summary_and_comparison_preserve_dataset_roles(self) -> None:
        rows = []
        for held_out_site, offset in (("ageri", 0.0), ("bianco", 0.1)):
            for dataset in ("train", "test", "loso"):
                row = {
                    "held_out_site": held_out_site,
                    "model": "random_forest",
                    "dataset": dataset,
                }
                for metric in (
                    "overall_binary_accuracy",
                    "balanced_binary_accuracy",
                    "total_cloud_accuracy",
                    "water_accuracy",
                    "total_cloud_user_accuracy",
                    "water_user_accuracy",
                    "total_cloud_f1",
                    "water_f1",
                ):
                    row[metric] = 0.7 + offset
                rows.append(row)
        detailed = pd.DataFrame(rows)
        summary = _summarize_binary_results(detailed)
        test_summary = summary.loc[summary["dataset"].eq("test")].iloc[0]
        self.assertEqual(test_summary["n_sites"], 2)
        self.assertAlmostEqual(test_summary["total_cloud_accuracy_mean"], 0.75)

        comparison = _binary_comparison_table(detailed)
        self.assertEqual(len(comparison), 2)
        self.assertIn("train_total_cloud_accuracy", comparison.columns)
        self.assertIn("test_water_accuracy", comparison.columns)
        self.assertIn("loso_overall_binary_accuracy", comparison.columns)


if __name__ == "__main__":
    unittest.main()
