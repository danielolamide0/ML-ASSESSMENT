"""Score new cases with a saved model.

Usage: python -m severity_triage.predict --input data/holdback_dataset.csv --output predictions/holdback_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from . import config as C
from .data import clean, load_raw
from .explain import top_drivers
from .model import SeverityModel

log = logging.getLogger("severity_triage.predict")


def load_model(path: Path = C.MODEL_PATH) -> SeverityModel:
    model = joblib.load(path)
    if not isinstance(model, SeverityModel):
        raise TypeError(f"{path} does not contain a SeverityModel")
    return model


def predict_dataframe(model: SeverityModel, raw: pd.DataFrame, *, explain: bool = False) -> tuple[pd.DataFrame, dict]:
    """Clean raw input, score it and return (predictions, data-quality report)."""
    cleaned, report = clean(raw, is_training=False)
    preds = model.predict_frame(cleaned)
    if explain:
        preds["TopDrivers"] = top_drivers(model, cleaned).values
    return preds, report.as_dict()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=C.HOLDBACK_PATH)
    parser.add_argument("--output", type=Path, default=C.PREDICTIONS_DIR / "holdback_predictions.csv")
    parser.add_argument("--model-path", type=Path, default=C.MODEL_PATH)
    parser.add_argument("--explain", action="store_true", help="add a TopDrivers column with the top SHAP reasons per case")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)

    model = load_model(args.model_path)
    log.info("Model trained %s (git %s), threshold %.2f", model.metadata.get("trained_at"), model.metadata.get("git_sha"), model.threshold)
    preds, report = predict_dataframe(model, load_raw(args.input), explain=args.explain)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    (args.output.with_suffix(".quality.json")).write_text(json.dumps(report, indent=2))
    summary = preds["PredictedSeverityScore"].value_counts().sort_index().to_dict()
    log.info("Wrote %d predictions to %s; score distribution %s; progressed %.1f%%", len(preds), args.output, summary, 100 * (preds["TriageDecision"] == "Progress").mean())


if __name__ == "__main__":
    main()
