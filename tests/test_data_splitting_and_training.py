from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from lswt_cloud_masking.data_splitting import SplitConfig, validate_splits
from lswt_cloud_masking.model_training import (
    TrainingConfig,
    _candidate_evaluation,
    _merge_candidate_records,
    _prepare_training_data,
    _sort_candidate_ranking,
    _weighted_harmonic_mean,
)
from lswt_cloud_masking.training_sensitivity import (
    SensitivityConfig,
    _assert_generated_run_paths,
    _los_overlap_table,
    _summarize_balanced_accuracy,
    _validate_sensitivity_config,
)


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_SCENE = "LC08_L1TP_196028_20210906_20210915_02_T1"


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_data = json.loads(
            (ROOT / "configs" / "training_split.example.json").read_text(encoding="utf-8")
        )
        cls.config = SplitConfig(**config_data)
        cls.frames = {
            "train": pd.read_csv(ROOT / cls.config.train_csv),
            "test": pd.read_csv(ROOT / cls.config.test_csv),
            "los_test": pd.read_csv(ROOT / cls.config.los_test_csv),
        }

    def test_split_sizes_coverage_and_los_exclusivity(self) -> None:
        validation = validate_splits(self.frames, self.config)
        total_rows = sum(
            len(pd.read_csv(ROOT / path)) for path in self.config.input_csvs
        )
        self.assertEqual(
            {name: len(frame) for name, frame in self.frames.items()},
            {
                "train": total_rows - round(total_rows * 0.1) * 2,
                "test": round(total_rows * 0.1),
                "los_test": round(total_rows * 0.1),
            },
        )
        self.assertEqual(validation["scene_overlap"]["train_los_test"]["n_overlap"], 0)
        self.assertEqual(validation["scene_overlap"]["test_los_test"]["n_overlap"], 0)
        for coverage in validation["coverage"].values():
            self.assertEqual(coverage["classes"], [1, 2, 3])
            self.assertEqual(coverage["sensors"], ["LC08", "LC09"])
            self.assertEqual(set(coverage["seasons"]), {"DJF", "MAM", "JJA", "SON"})
        for name in ("train", "test"):
            bianco_classes = set(
                self.frames[name].loc[self.frames[name]["lakeN"].eq("bianco"), "lst_class"]
            )
            self.assertEqual(bianco_classes, {1.0, 2.0, 3.0})

    def test_protected_scene_has_configured_development_allocation(self) -> None:
        source = pd.concat(
            [
                pd.read_csv(ROOT / path)
                for path in self.config.input_csvs
            ],
            ignore_index=True,
        )
        source_count = int(source["scene_id"].eq(PROTECTED_SCENE).sum())
        self.assertGreater(source_count, 0)
        self.assertEqual(
            int(self.frames["train"]["scene_id"].eq(PROTECTED_SCENE).sum()),
            120,
        )
        self.assertEqual(
            int(self.frames["test"]["scene_id"].eq(PROTECTED_SCENE).sum()),
            30,
        )
        self.assertFalse(self.frames["los_test"]["scene_id"].eq(PROTECTED_SCENE).any())

        output = pd.concat(self.frames.values(), ignore_index=True)
        source_hashes = pd.util.hash_pandas_object(source, index=False).value_counts().sort_index()
        output_hashes = pd.util.hash_pandas_object(output, index=False).value_counts().sort_index()
        self.assertTrue(source_hashes.equals(output_hashes), "Split output lost or duplicated rows.")


class TrainingPipelineTests(unittest.TestCase):
    def test_training_data_has_disjoint_los_scenes_and_aligned_features(self) -> None:
        config = TrainingConfig(
            train_csv=str(ROOT / "data" / "df_train_80.csv"),
            test_csv=str(ROOT / "data" / "df_test_inscene_10.csv"),
            los_test_csv=str(ROOT / "data" / "df_test_los_10.csv"),
            output_dir=str(ROOT / "models" / "general" / "80_10_10"),
        )
        data = _prepare_training_data(config)
        self.assertEqual(list(data.x_train.columns), list(data.x_test.columns))
        self.assertEqual(list(data.x_train.columns), list(data.x_los_test.columns))
        self.assertNotIn("scene_id", data.feature_columns)
        self.assertFalse(set(data.train_groups) & set(data.los_test_groups))
        self.assertFalse(set(data.test_groups) & set(data.los_test_groups))

    def test_xgb_pareto_selection_score_balances_both_objectives(self) -> None:
        score = _weighted_harmonic_mean(0.6, 0.9, 0.5)
        self.assertAlmostEqual(score, 0.72)
        self.assertLess(score, 0.9)

    def test_repeated_candidate_selection_uses_stability_adjusted_mean(self) -> None:
        config = TrainingConfig(
            train_csv="train.csv",
            test_csv="test.csv",
            output_dir="models",
            stability_penalty=0.25,
        )
        candidates = [
            _candidate_evaluation(
                {"candidate_id": 1, "parameters": {"max_depth": 10}, "sources": []},
                [1.0, 0.6],
                {"42": [1.0], "142": [0.6]},
                config,
            ),
            _candidate_evaluation(
                {"candidate_id": 2, "parameters": {"max_depth": 5}, "sources": []},
                [0.76, 0.76],
                {"42": [0.76], "142": [0.76]},
                config,
            ),
        ]
        ranking = _sort_candidate_ranking(candidates)
        self.assertEqual(ranking[0]["candidate_id"], 2)
        self.assertAlmostEqual(ranking[0]["stability_score"], 0.76)
        self.assertAlmostEqual(ranking[1]["stability_score"], 0.75)

    def test_duplicate_candidate_parameters_merge_across_runs(self) -> None:
        records = [
            {
                "parameters": {"max_depth": 5, "criterion": "gini"},
                "sources": [{"run": 1, "trial": 3}],
            },
            {
                "parameters": {"criterion": "gini", "max_depth": 5},
                "sources": [{"run": 2, "trial": 1}],
            },
        ]
        merged = _merge_candidate_records(records)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["sources"]), 2)


class SensitivityPipelineTests(unittest.TestCase):
    def test_balanced_accuracy_summary_aggregates_runs_by_model(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "model": "random_forest",
                    "train_balanced_accuracy": 0.90,
                    "test_inscene_balanced_accuracy": 0.80,
                    "los_balanced_accuracy": 0.60,
                },
                {
                    "model": "random_forest",
                    "train_balanced_accuracy": 0.94,
                    "test_inscene_balanced_accuracy": 0.84,
                    "los_balanced_accuracy": 0.70,
                },
            ]
        )
        summary = _summarize_balanced_accuracy(results).iloc[0]
        self.assertEqual(summary["n_runs"], 2)
        self.assertAlmostEqual(summary["train_balanced_accuracy_mean"], 0.92)
        self.assertAlmostEqual(summary["test_inscene_balanced_accuracy_mean"], 0.82)
        self.assertAlmostEqual(summary["los_balanced_accuracy_mean"], 0.65)
        self.assertGreater(summary["los_balanced_accuracy_std"], 0)

    def test_los_overlap_table_reports_scene_reuse_across_splits(self) -> None:
        overlap = _los_overlap_table(
            {
                "run_01": {"scene_a", "scene_b"},
                "run_02": {"scene_b", "scene_c"},
            }
        )
        between = overlap.loc[
            overlap["run_a"].eq("run_01") & overlap["run_b"].eq("run_02")
        ].iloc[0]
        self.assertEqual(between["overlap_scene_count"], 1)
        self.assertEqual(between["union_scene_count"], 3)
        self.assertAlmostEqual(between["jaccard_similarity"], 1 / 3)

    def test_sensitivity_split_seeds_must_be_distinct(self) -> None:
        config = SensitivityConfig(
            split_config="split.json",
            training_config="training.json",
            output_dir="outputs",
            split_seeds=[42, 42],
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            _validate_sensitivity_config(config)

    def test_sensitivity_rejects_dataset_path_overrides(self) -> None:
        config = SensitivityConfig(
            split_config="split.json",
            training_config="training.json",
            output_dir="outputs",
            split_seeds=[42],
            training_overrides={"train_csv": "data/df_train_80.csv"},
        )
        with self.assertRaisesRegex(ValueError, "generated per run"):
            _validate_sensitivity_config(config)

    def test_training_paths_must_match_generated_split_paths(self) -> None:
        dataset_dir = Path("/tmp/sensitivity/datasets/run_01_seed_42")
        model_dir = Path("/tmp/sensitivity/models/run_01_seed_42")
        split = SplitConfig(
            input_csvs=["df_train_all.csv", "df_test_all.csv"],
            train_csv=str(dataset_dir / "df_train_80.csv"),
            test_csv=str(dataset_dir / "df_test_inscene_10.csv"),
            los_test_csv=str(dataset_dir / "df_test_los_10.csv"),
            manifest_csv=str(dataset_dir / "manifest.csv"),
            summary_json=str(dataset_dir / "summary.json"),
        )
        training = TrainingConfig(
            train_csv=split.train_csv,
            test_csv=split.test_csv,
            los_test_csv=split.los_test_csv,
            output_dir=str(model_dir),
        )
        _assert_generated_run_paths(split, training, dataset_dir, model_dir)

        training.train_csv = "data/df_train_80.csv"
        with self.assertRaisesRegex(RuntimeError, "generated dataset"):
            _assert_generated_run_paths(split, training, dataset_dir, model_dir)


if __name__ == "__main__":
    unittest.main()
