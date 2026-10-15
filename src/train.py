"""Fit an unsupervised Isolation Forest and prepare a review queue."""

import json
from pathlib import Path
import shutil

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data import CATEGORICAL_FEATURES, DATA_PATH, FEATURES, NUMERIC_FEATURES, load_or_generate


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
DOC_ASSETS = DOCS / "assets"


def main() -> None:
    frame = load_or_generate(force=True)
    preprocess = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numbers", RobustScaler(), NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("detector", IsolationForest(n_estimators=400, contamination=0.045, random_state=27, n_jobs=-1)),
        ]
    )
    model.fit(frame[FEATURES])
    anomaly_score = -model.decision_function(frame[FEATURES])
    threshold = 0.0
    flagged = anomaly_score > threshold
    scored = frame.copy()
    scored["anomaly_score"] = anomaly_score.round(6)
    scored["review_priority"] = np.where(flagged, "review", "routine")
    scored = scored.sort_values("anomaly_score", ascending=False)
    review_report = scored.loc[scored["review_priority"] == "review"].copy()
    metrics = {
        "dataset": "synthetic_employee_expense_claims",
        "rows": int(len(frame)),
        "flagged_claims": int(flagged.sum()),
        "flagged_rate": float(flagged.mean()),
        "contamination_setting": 0.045,
        "anomaly_score_threshold": threshold,
        "evaluation_note": "No ground-truth fraud labels; accuracy, precision, recall, and F1 are not available.",
    }
    ARTIFACTS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    DOC_ASSETS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "employee_expense_isolation_forest.joblib")
    metrics_path = ARTIFACTS / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scored_path = ARTIFACTS / "scored_claims.csv"
    report_path = ARTIFACTS / "suspicious_claim_report.csv"
    scored.to_csv(scored_path, index=False)
    review_report.to_csv(report_path, index=False)
    downloads = DOCS / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, DOCS / "metrics.json")
    shutil.copy2(metrics_path, downloads / "metrics.json")
    shutil.copy2(scored_path, downloads / "scored_claims.csv")
    shutil.copy2(report_path, downloads / "suspicious_claim_report.csv")
    shutil.copy2(DATA_PATH, downloads / DATA_PATH.name)

    fig, axis = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    axis.hist(scored["anomaly_score"], bins=36, color="#9bc5bd", edgecolor="#f3f1e9", linewidth=0.8)
    axis.axvline(metrics["anomaly_score_threshold"], color="#d5533f", linewidth=2, label="Review threshold")
    axis.set(title="Claim anomaly-score distribution", xlabel="Anomaly score · higher is more unusual", ylabel="Claims")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(DOC_ASSETS / "anomaly-distribution.png", dpi=170, facecolor="#f4f4ee")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"Model: {ARTIFACTS / 'employee_expense_isolation_forest.joblib'}")
    print(f"Review report: {ARTIFACTS / 'suspicious_claim_report.csv'}")
    print(f"Data: {DATA_PATH}")


if __name__ == "__main__":
    main()