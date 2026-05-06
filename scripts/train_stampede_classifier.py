"""Train a lightweight stampede signal classifier.

Per improvement plan Area 4: train a shallow classifier on top of the 5
Farneback flow signals. The Sciencedirect paper S0952197624020992 reports
~99% on UMN/PETS-2009 using this approach with a deep classifier.

Inputs (5 features per frame):
  density_norm    crowd.max_density / danger_density  (clamped to [0, 2])
  divergence      flow.divergence
  convergence     flow.convergence
  variance_norm   min(flow.variance / 15.0, 2.0)
  counter_flow    flow.counter_flow

Output labels: "ok" | "warning" | "critical"

Usage:
  # 1. Run pipeline on labeled footage and export signal CSV:
  python scripts/export_stampede_signals.py --source data/umn_videos --out data/signals.csv

  # 2. Train the classifier:
  python scripts/train_stampede_classifier.py \
      --csv data/signals.csv --out models/stampede_clf.pkl

  # 3. Load at runtime in StampedeDetector:
  stampede_det.load_classifier("models/stampede_clf.pkl")

  Alternatively, set `stampede_classifier_path` in rules.yaml and the
  pipeline will load it automatically on startup.

Datasets to use (public, per improvement plan):
  - UMN crowd dataset (5 sequences)
  - PETS-2009 (multi-camera crowd)
  - GBA-Stampedes + GSMADC from Sciencedirect S0952197624020992 (~43,000 frames)
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

FEATURES = ["density_norm", "divergence", "convergence", "variance_norm", "counter_flow"]
LABEL = "label"


def load_csv(csv_path: str):
    import csv
    X, y = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = [float(row[feat]) for feat in FEATURES]
                label = row[LABEL].strip().lower()
                if label not in ("ok", "warning", "critical"):
                    continue
                X.append(x)
                y.append(label)
            except (KeyError, ValueError):
                continue
    return np.array(X, dtype=np.float32), np.array(y)


def train(X, y, model_type: str = "gradient_boosting"):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    if model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # 5-fold cross-validation
    scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
    print(f"Cross-val F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")

    pipe.fit(X, y)
    print(f"Training accuracy: {pipe.score(X, y):.3f}")
    print(f"Classes: {list(pipe.classes_)}")
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Signal CSV with columns: " + ", ".join(FEATURES + [LABEL]))
    parser.add_argument("--out", default="models/stampede_clf.pkl")
    parser.add_argument("--model", default="gradient_boosting", choices=["gradient_boosting", "mlp", "random_forest"])
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    X, y = load_csv(args.csv)
    print(f"Loaded {len(X)} samples. Label counts: { {l: (y==l).sum() for l in set(y)} }")

    if len(X) < 50:
        print("ERROR: Need at least 50 samples. Generate more labeled data first.")
        return

    pipe = train(X, y, model_type=args.model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved classifier to {out_path}")
    print(f"Load in pipeline: stampede_det.load_classifier('{out_path}')")


if __name__ == "__main__":
    main()
