from pathlib import Path

import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL_NO = "BT2024267"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PRED_DIR = ROOT / "predictions"


def load(split, var):
    df = pd.read_csv(DATA_DIR / f"{ROLL_NO}_{split}_var{var}.csv")
    cols = [c for c in df.columns if c.startswith("x")]
    y = None
    if "y" in df.columns:
        y = df["y"].to_numpy()
    return df[cols].to_numpy(), y


def build_model(degree, method="ols", alpha=0.0):
    if method == "ols":
        reg = LinearRegression()
    elif method == "ridge":
        reg = Ridge(alpha=alpha)
    elif method == "lasso":
        reg = Lasso(alpha=alpha, max_iter=50000, tol=1e-5)
    else:
        raise ValueError("unknown method " + method)

    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        reg,
    )
