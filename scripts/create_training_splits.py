from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lswt_cloud_masking.data_splitting import SplitConfig, run_split_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible training, familiar-test, and leave-out-of-scene CSVs."
    )
    parser.add_argument(
        "--config",
        default="configs/training_split.example.json",
        help="Split configuration JSON. Default: configs/training_split.example.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = SplitConfig.from_json(config_path)

    for field_name in [
        "input_csvs",
        "train_csv",
        "test_csv",
        "los_test_csv",
        "manifest_csv",
        "summary_json",
    ]:
        value = getattr(config, field_name)
        if isinstance(value, list):
            setattr(config, field_name, [str((ROOT / item).resolve()) for item in value])
        else:
            setattr(config, field_name, str((ROOT / value).resolve()))

    result = run_split_pipeline(config)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
