"""Run raw CSV batch inference with the selected credit and fraud bundles."""

from __future__ import annotations

import argparse

from src.pipelines.inference_pipeline import run_batch_inference
from src.utils.config import load_config


def _arguments() -> argparse.Namespace:
    """Parse explicit input/output and model-version controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Raw customer CSV; default comes from config.")
    parser.add_argument("--output", help="Prediction CSV; default comes from config.")
    parser.add_argument("--task", choices=("credit", "fraud", "both"), default="both")
    parser.add_argument("--version", help="Bundle version; default comes from config.")
    parser.add_argument("--model-root", default="models")
    return parser.parse_args()


def main() -> None:
    """Run one configured raw-batch inference job."""
    arguments = _arguments()
    config = load_config()
    settings = config["inference"]
    run_batch_inference(
        arguments.input or settings["input_path"],
        arguments.output or settings["output_path"],
        task=arguments.task,
        version=arguments.version or settings["model_version"],
        model_root=arguments.model_root,
    )


if __name__ == "__main__":
    main()
