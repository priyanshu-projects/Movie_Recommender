"""
scripts/retrain_svd.py
"""
import json
import sys
from pathlib import Path
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.svd_model import SVDModel
from src.data.temporal_split import global_temporal_split

def main():
    print("🚀 Retraining SVD Model...")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    ratings_path = Path("data/processed/all_ratings.csv")
    if not ratings_path.exists():
        print(f"❌ Ratings file not found at {ratings_path}")
        sys.exit(1)

    ratings = pd.read_csv(ratings_path)
    train, val, test = global_temporal_split(ratings)
    model = SVDModel(cfg["svd"])
    model.train(train)
    Path("models").mkdir(exist_ok=True)
    model.save(Path("models/svd_candidate.pkl"))
    metrics = model.evaluate(test, k=cfg["evaluation"]["k"])
    print(f"✓ SVD trained | metrics: {metrics}")

    with open("models/svd_metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
