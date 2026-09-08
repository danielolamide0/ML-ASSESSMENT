"""Train the severity model with cross-validation and save the artefact.

Usage: python -m severity_triage.train [--exclude-internal-scores] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import __version__
from . import config as C
from .data import load_and_clean, split_xy
from .evaluate import choose_threshold, full_evaluation
from .model import SeverityModel, build_baseline_pipeline, build_lgbm_pipeline

log = logging.getLogger("severity_triage.train")


def cross_validate_proba(pipeline, X: pd.DataFrame, y: pd.Series, n_folds: int, seed: int) -> np.ndarray:  # noqa: N803
    """Out-of-fold class probabilities (columns ordered by C.SEVERITY_CLASSES)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), len(C.SEVERITY_CLASSES)))
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        est = clone(pipeline)
        est.fit(X.iloc[tr], y.iloc[tr])
        proba = est.predict_proba(X.iloc[va])
        order = [list(est.named_steps["model"].classes_).index(c) for c in C.SEVERITY_CLASSES]
        oof[va] = proba[:, order]
        log.info("fold %d/%d done", fold, n_folds)
    return oof


def _git_sha() -> str | None:
    """Short SHA of HEAD, suffixed with '-dirty' if the working tree differs from it."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=C.PROJECT_ROOT).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "pyproject.toml"], cwd=C.PROJECT_ROOT).returncode != 0
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001
        return None


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def train(cfg: C.TrainingConfig, train_path: Path = C.TRAIN_PATH, model_path: Path = C.MODEL_PATH) -> tuple[SeverityModel, dict]:
    np.random.seed(cfg.seed)
    df, quality = load_and_clean(train_path, is_training=True)
    X, y = split_xy(df)

    results: dict = {"n_train": int(len(X)), "class_distribution": y.value_counts().sort_index().to_dict()}

    # 1. Baseline for reference
    baseline = build_baseline_pipeline(exclude_internal_scores=cfg.exclude_internal_scores, seed=cfg.seed)
    oof_base = cross_validate_proba(baseline, X, y, cfg.n_folds, cfg.seed)
    t_base, _ = choose_threshold(y, oof_base, cfg.costs, cfg.min_severe_recall)
    results["baseline_logistic_cv"] = full_evaluation(y, oof_base, t_base, cfg.costs)

    # 2. Main model
    pipeline = build_lgbm_pipeline(cfg.lgbm_params, exclude_internal_scores=cfg.exclude_internal_scores, seed=cfg.seed)
    oof = cross_validate_proba(pipeline, X, y, cfg.n_folds, cfg.seed)
    threshold, curve = choose_threshold(y, oof, cfg.costs, cfg.min_severe_recall)
    results["lightgbm_cv"] = full_evaluation(y, oof, threshold, cfg.costs)
    results["threshold_curve"] = curve.to_dict(orient="records")

    # 3. Final fit on all data
    pipeline.fit(X, y)
    metadata = {
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "training_data": str(train_path.relative_to(C.PROJECT_ROOT)) if train_path.is_relative_to(C.PROJECT_ROOT) else str(train_path),
        "training_data_sha256": _file_hash(train_path),
        "n_train": int(len(X)),
        "seed": cfg.seed,
        "n_folds": cfg.n_folds,
        "exclude_internal_scores": cfg.exclude_internal_scores,
        "costs": cfg.costs.__dict__,
        "min_severe_recall": cfg.min_severe_recall,
        "lgbm_params": cfg.lgbm_params,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "data_quality": quality.as_dict(),
        "cv_metrics": {k: results["lightgbm_cv"][k] for k in ("score", "triage", "probability")},
    }
    model = SeverityModel(pipeline=pipeline, threshold=threshold, metadata=metadata)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    (model_path.parent / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    (model_path.parent / "model_card.json").write_text(json.dumps(metadata, indent=2, default=str))
    np.save(model_path.parent / "oof_proba.npy", oof)
    log.info("Saved model to %s (threshold=%.2f)", model_path, threshold)
    return model, results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=C.TRAIN_PATH)
    parser.add_argument("--model-path", type=Path, default=C.MODEL_PATH)
    parser.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--exclude-internal-scores", action="store_true")
    parser.add_argument("--fn-cost", type=float, default=C.BusinessCosts().false_negative)
    parser.add_argument("--fp-cost", type=float, default=C.BusinessCosts().false_positive)
    parser.add_argument("--min-severe-recall", type=float, default=0.95)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)

    cfg = C.TrainingConfig(
        seed=args.seed,
        n_folds=args.folds,
        exclude_internal_scores=args.exclude_internal_scores,
        costs=C.BusinessCosts(args.fn_cost, args.fp_cost),
        min_severe_recall=args.min_severe_recall,
    )
    _, results = train(cfg, args.train_path, args.model_path)
    cv = results["lightgbm_cv"]
    print(json.dumps({"threshold": cv["threshold"], "score": cv["score"], "triage": cv["triage"], "probability": cv["probability"]}, indent=2))


if __name__ == "__main__":
    main()
