import subprocess
import sys

import pandas as pd

from severity_triage import config as C


def test_predict_cli_end_to_end(tmp_path):
    """The committed model scores the holdback file: 500 unique cases, consistent decisions."""
    out = tmp_path / "preds.csv"
    subprocess.run(
        [sys.executable, "-m", "severity_triage.predict", "--input", str(C.HOLDBACK_PATH), "--output", str(out)],
        check=True,
        capture_output=True,
    )
    preds = pd.read_csv(out)
    holdback = pd.read_csv(C.HOLDBACK_PATH)
    assert len(preds) == len(holdback) == 500
    assert preds[C.ID_COLUMN].is_unique
    assert set(preds[C.ID_COLUMN]) == set(holdback[C.ID_COLUMN])
    assert preds["PredictedSeverityScore"].isin(C.SEVERITY_CLASSES).all()
    assert ((preds["PredictedSeverityScore"] >= 4) == (preds["TriageDecision"] == "Progress")).all()
    assert out.with_suffix(".quality.json").exists()
