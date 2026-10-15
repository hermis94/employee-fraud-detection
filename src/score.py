"""Rank employee expense claims for investigation with a saved Isolation Forest."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import FEATURES


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "employee_expense_isolation_forest.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV containing all claim feature columns")
    parser.add_argument("--output", type=Path, default=Path("suspicious_claim_report.csv"))
    parser.add_argument("--all-scored", type=Path, default=None, help="Optional output path for every scored claim")
    arguments = parser.parse_args()
    frame = pd.read_csv(arguments.input_csv)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        parser.error(f"input is missing required feature columns: {', '.join(missing)}")
    if not MODEL_PATH.exists():
        parser.error("model is missing; run `python -m src.train` first")
    model = joblib.load(MODEL_PATH)
    score = -model.decision_function(frame[FEATURES])
    threshold = 0.0
    result = frame.copy()
    result["anomaly_score"] = score
    result["review_priority"] = np.where(score > threshold, "review", "routine")
    result = result.sort_values("anomaly_score", ascending=False)
    result[result["review_priority"] == "review"].to_csv(arguments.output, index=False)
    if arguments.all_scored:
        result.to_csv(arguments.all_scored, index=False)
    print(f"Flagged {result['review_priority'].eq('review').sum()} of {len(result)} claims for human review.")
    print(f"Wrote ranked review queue to {arguments.output}")


if __name__ == "__main__":
    main()