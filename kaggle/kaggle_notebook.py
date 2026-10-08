import glob
import os

import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL_NO = "BT2024267"
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working")
RUN_CV_CHECK = True


def get_model(var):
    if var == 1:
        degree = 5
        reg = Lasso(alpha=0.01, max_iter=50000, tol=1e-5)
    else:
        degree = 10
        reg = Ridge(alpha=1.0)
    return make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), reg)


def find_file(kind, var):
    all_csv = glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True)
    matches = []
    for p in all_csv:
        name = os.path.basename(p).lower()
        if kind in name and f"var{var}" in name:
            matches.append(p)
    if not matches:
        return None
    wanted = f"{ROLL_NO}_{kind}_var{var}.csv".lower()
    for p in matches:
        if os.path.basename(p).lower() == wanted:
            return p
    return sorted(matches)[0]


print("CSV files under", INPUT_DIR)
for p in sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True)):
    print("  ", p)
print()

os.makedirs(OUTPUT_DIR, exist_ok=True)

for var in (1, 2):
    train_path = find_file("train", var)
    test_path = find_file("test", var)
    if train_path is None:
        print(f"var{var}: no training file found\n")
        continue

    train = pd.read_csv(train_path)
    cols = sorted([c for c in train.columns if c.lower().startswith("x")], key=lambda c: int(c[1:]))
    X = train[cols].to_numpy()
    y = train["y"].to_numpy()
    print(f"var{var} train: {train_path} shape={train.shape} features={cols}")

    model = get_model(var)
    if RUN_CV_CHECK:
        scores = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0),
                                 scoring="neg_mean_squared_error")
        mse = -scores.mean()
        print(f"var{var} 5-fold CV  MSE={mse:.4f} R2={1 - mse / y.var():.4f}")

    model.fit(X, y)
    p = model.predict(X)
    print(f"var{var} train fit  MSE={mean_squared_error(y, p):.4f} R2={r2_score(y, p):.4f}")

    if test_path is None:
        print(f"var{var}: no test file found\n")
        continue

    test = pd.read_csv(test_path)
    missing = [c for c in cols if c not in test.columns]
    if missing:
        raise ValueError(f"{test_path} is missing columns {missing}")
    pred = model.predict(test[cols].to_numpy())

    sub = pd.DataFrame({"y": pred})
    out = os.path.join(OUTPUT_DIR, f"{ROLL_NO}_pred_var{var}.csv")
    sub.to_csv(out, index=False)
    print(f"var{var} test: {test_path} rows={len(test)} -> {out}")
    if "y" in test.columns:
        print(f"var{var} TEST MSE={mean_squared_error(test['y'], pred):.4f} "
              f"R2={r2_score(test['y'], pred):.4f}")
    print(sub.head(), "\n")

print("Done. Files:", sorted(os.listdir(OUTPUT_DIR)))
