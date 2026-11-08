"""Train Giai đoạn 6 LightGBM models; use --final only for the one test evaluation."""

from __future__ import annotations

import argparse

from src.pipelines.training_pipeline import run_training
from src.utils.config import load_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse main-model execution controls without exposing arbitrary config mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="Evaluate the held-out test split once.")
    parser.add_argument("--force", action="store_true", help="Allow a deliberate repeat of test evaluation.")
    parser.add_argument(
        "--smoke", action="store_true", help="Use small train/tuning samples and write models under smoke/."
    )
    return parser.parse_args()


def main() -> None:
    """Run configured training or a small local smoke check."""
    arguments = parse_args()
    if arguments.final and arguments.smoke:
        raise ValueError("--final cannot be combined with --smoke.")
    result = run_training(
        load_config(),
        final=arguments.final,
        force=arguments.force,
        smoke_train_size=20_000 if arguments.smoke else None,
        smoke_tune_size=5_000 if arguments.smoke else None,
        smoke_trials=2 if arguments.smoke else None,
        smoke_timeout_seconds=120 if arguments.smoke else None,
    )
    LOGGER.info("Main-model training result: %s", result)


if __name__ == "__main__":
    main()
