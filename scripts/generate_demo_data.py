from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "customer_churn_demo.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
X, y = make_classification(n_samples=600, n_features=8, n_informative=5, n_redundant=1,
                           weights=[0.72, 0.28], random_state=42)
rng = np.random.default_rng(42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 9)])
df["monthly_plan"] = rng.choice(["basic", "standard", "premium"], size=len(df), p=[0.4, 0.4, 0.2])
df["support_channel"] = rng.choice(["email", "phone", "chat"], size=len(df))
df["churn"] = np.where(y == 1, "yes", "no")
for col in ["feature_2", "feature_5"]:
    mask = rng.random(len(df)) < 0.04
    df.loc[mask, col] = np.nan
df.to_csv(OUT, index=False)
print(f"Demo dataset saved to: {OUT}")
