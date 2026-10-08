import argparse
import json
from math import comb

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

from common import RESULTS_DIR, build_model, load

N_SPLITS = 5
N_REPEATS = 3
SEED = 42

GRID = {
    1: {
        "degrees": range(1, 8),
        "ridge": [0.01, 0.1, 1, 3, 10, 30, 100],
        "lasso": [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03],
        "lasso_degrees": range(3, 8),
    },
    2: {
        "degrees": range(1, 21),
        "ridge": [1e-4, 1e-3, 0.01, 0.1, 1, 10],
        "lasso": [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01],
        "lasso_degrees": range(5, 15),
    },
}


def n_terms(n_vars, degree):
    return comb(n_vars + degree, degree) - 1


def get_configs(var, n_vars, n_train):
    grid = GRID[var]
    configs = []
    for d in grid["degrees"]:
        if n_terms(n_vars, d) < 0.8 * n_train:
            configs.append((d, "ols", 0.0))
        for a in grid["ridge"]:
            configs.append((d, "ridge", a))
    for d in grid["lasso_degrees"]:
        for a in grid["lasso"]:
            configs.append((d, "lasso", a))
    return configs


def score_fold(X, y, train_idx, val_idx, degree, method, alpha):
    model = build_model(degree, method, alpha)
    model.fit(X[train_idx], y[train_idx])
    return mean_squared_error(y[val_idx], model.predict(X[val_idx]))


def run(var):
    X, y = load("train", var)
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(cv.split(X))
    configs = get_configs(var, X.shape[1], len(folds[0][0]))
    print(f"var{var}: {len(configs)} configs, {len(folds)} folds each")

    jobs = []
    for c in configs:
        for tr, te in folds:
            jobs.append(delayed(score_fold)(X, y, tr, te, *c))
    scores = Parallel(n_jobs=-1)(jobs)
    scores = np.array(scores).reshape(len(configs), len(folds))

    df = pd.DataFrame(configs, columns=["degree", "method", "alpha"])
    df["n_terms"] = [n_terms(X.shape[1], d) for d in df.degree]
    df["cv_mse"] = scores.mean(axis=1)
    df["cv_mse_se"] = scores.std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
    df["cv_r2"] = 1 - df.cv_mse / y.var()
    df = df.sort_values("cv_mse").reset_index(drop=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"cv_var{var}.csv", index=False)

    best = {}
    for k, v in df.iloc[0].to_dict().items():
        best[k] = v.item() if hasattr(v, "item") else v
    with open(RESULTS_DIR / f"best_var{var}.json", "w") as f:
        json.dump(best, f, indent=2)

    per_degree = df.loc[df.groupby(["method", "degree"]).cv_mse.idxmin()]
    per_degree = per_degree.sort_values(["method", "degree"])
    print(per_degree[["degree", "method", "alpha", "n_terms", "cv_mse", "cv_r2"]].to_string(index=False))
    print(f"var{var} best: {best}\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=int, choices=[1, 2], nargs="*", default=[1, 2])
    args = parser.parse_args()
    for v in args.var:
        run(v)
