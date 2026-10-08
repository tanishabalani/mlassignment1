import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.polynomial import legendre
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import RESULTS_DIR, build_model, load

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42

ALTERNATIVES = {
    1: [(4, "lasso", 0.0075), (5, "ridge", 30.0), (6, "lasso", 0.01)],
    2: [(8, "ols", 0.0), (8, "ridge", 0.1), (12, "ridge", 1.0)],
}


class LegendreFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y=None):
        self.powers_ = PolynomialFeatures(self.degree, include_bias=False).fit(X).powers_
        return self

    def transform(self, X):
        P = []
        for k in range(self.degree + 1):
            coef = np.zeros(self.degree + 1)
            coef[k] = 1
            P.append(legendre.legval(X, coef))
        P = np.stack(P)
        F = np.ones((X.shape[0], len(self.powers_)))
        for j in range(X.shape[1]):
            F *= P[self.powers_[:, j], :, j].T
        return F


class RelaxedLasso(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        lasso = Lasso(alpha=self.alpha, max_iter=50000).fit(X, y)
        self.support_ = np.flatnonzero(lasso.coef_)
        self.ols_ = LinearRegression().fit(X[:, self.support_], y)
        return self

    def predict(self, X):
        return self.ols_.predict(X[:, self.support_])


def make_folds(X):
    return list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X))


def oof_errors(model_fn, X, y, folds):
    repeats = len(folds) // 5
    err = np.zeros((repeats, len(y)))
    for i, (tr, te) in enumerate(folds):
        model = model_fn().fit(X[tr], y[tr])
        err[i // 5, te] = (model.predict(X[te]) - y[te]) ** 2
    return err.mean(axis=0)


def count_at_bound(X):
    return (np.abs(X) == 1).sum(axis=1)


def train_mse(cfg, X, y):
    model = build_model(*cfg).fit(X, y)
    return float(np.mean((model.predict(X) - y) ** 2))


class ModelFactory:
    def __init__(self, kind, degree, alpha):
        self.kind = kind
        self.degree = degree
        self.alpha = alpha

    def __call__(self):
        if self.kind == "Legendre basis + Ridge":
            return make_pipeline(LegendreFeatures(self.degree), StandardScaler(), Ridge(alpha=self.alpha))
        if self.kind == "Legendre basis + Lasso":
            return make_pipeline(LegendreFeatures(self.degree), StandardScaler(),
                                 Lasso(alpha=self.alpha, max_iter=50000))
        if self.kind == "Relaxed Lasso (monomials)":
            return make_pipeline(PolynomialFeatures(self.degree, include_bias=False), StandardScaler(),
                                 RelaxedLasso(self.alpha))
        return build_model(self.degree, self.kind, self.alpha)


def other_techniques(degree):
    out = []
    for a in (0.1, 1.0, 10.0, 30.0):
        out.append(ModelFactory("Legendre basis + Ridge", degree, a))
    for a in (0.001, 0.005, 0.01):
        out.append(ModelFactory("Legendre basis + Lasso", degree, a))
    for a in (0.005, 0.01, 0.02, 0.04):
        out.append(ModelFactory("Relaxed Lasso (monomials)", degree, a))
    return out


def run_var(var, best):
    X, y = load("train", var)
    T, _ = load("test", var)
    folds = make_folds(X)
    sel = (int(best["degree"]), best["method"], float(best["alpha"]))
    out = {"selected": sel}

    out["describe"] = []
    for j in range(X.shape[1]):
        out["describe"].append({
            "var": f"x{j + 1}",
            "train_mean": float(X[:, j].mean()), "train_sd": float(X[:, j].std()),
            "test_mean": float(T[:, j].mean()), "test_sd": float(T[:, j].std()),
            "train_bound": float((np.abs(X[:, j]) == 1).mean()),
            "test_bound": float((np.abs(T[:, j]) == 1).mean()),
        })

    cv = pd.read_csv(RESULTS_DIR / f"cv_var{var}.csv")
    per_degree = cv.loc[cv.groupby(["method", "degree"]).cv_mse.idxmin()]
    cfgs = [(int(r.degree), r.method, float(r.alpha)) for r in per_degree.itertuples()]
    train_errs = Parallel(n_jobs=-1)(delayed(train_mse)(c, X, y) for c in cfgs)
    out["train_vs_cv"] = []
    for c, t, r in zip(cfgs, train_errs, per_degree.itertuples()):
        out["train_vs_cv"].append({"degree": c[0], "method": c[1], "alpha": c[2],
                                   "train_mse": t, "cv_mse": float(r.cv_mse)})

    n = X.shape[1]
    k_train = count_at_bound(X)
    k_test = count_at_bound(T)
    out["bound_counts"] = {"train": np.bincount(k_train, minlength=n + 1).tolist(),
                           "test": np.bincount(k_test, minlength=n + 1).tolist()}
    out["frac_values_at_bound"] = {"train": float((np.abs(X) == 1).mean()),
                                   "test": float((np.abs(T) == 1).mean())}
    test_weights = np.bincount(k_test, minlength=n + 1) / len(k_test)

    cfgs = [sel] + ALTERNATIVES[var]
    factories = [ModelFactory(c[1], c[0], c[2]) for c in cfgs]
    errs = Parallel(n_jobs=-1)(delayed(oof_errors)(f, X, y, folds) for f in factories)
    rows = []
    for c, e in zip(cfgs, errs):
        per_k = []
        for k in range(n + 1):
            if (k_train == k).any():
                per_k.append(float(e[k_train == k].mean()))
            else:
                per_k.append(None)
        filled = []
        last = None
        for v in per_k:
            if v is not None:
                last = v
            filled.append(last)
        rows.append({"config": list(c), "cv_mse": float(e.mean()), "per_k_mse": per_k,
                     "test_weighted_mse": float(np.dot(test_weights, filled))})
    out["shift_check"] = rows

    final = build_model(*sel).fit(X, y)
    pred = final.predict(T)
    w_mse = rows[0]["test_weighted_mse"]
    out["test_pred"] = {"mean": float(pred.mean()), "std": float(pred.std()),
                        "min": float(pred.min()), "max": float(pred.max())}
    out["train_y"] = {"mean": float(y.mean()), "std": float(y.std()),
                      "min": float(y.min()), "max": float(y.max())}
    out["expected_test"] = {"mse": w_mse, "r2": float(1 - w_mse / (pred.var() + w_mse))}

    factories = other_techniques(sel[0])
    errs = Parallel(n_jobs=-1)(delayed(oof_errors)(f, X, y, folds) for f in factories)
    tech = {}
    for f, e in zip(factories, errs):
        m = float(e.mean())
        if f.kind not in tech or m < tech[f.kind]["cv_mse"]:
            tech[f.kind] = {"alpha": f.alpha, "cv_mse": m}
    out["other_techniques"] = tech

    if sel[1] == "lasso":
        names = final[0].get_feature_names_out([f"x{i + 1}" for i in range(n)])
        coef = final[-1].coef_
        nz = np.flatnonzero(coef)
        degs = final[0].powers_.sum(axis=1)
        out["lasso_terms_by_degree"] = {}
        for d in range(1, sel[0] + 1):
            out["lasso_terms_by_degree"][d] = [int((degs[nz] == d).sum()), int((degs == d).sum())]
        top = nz[np.argsort(-np.abs(coef[nz]))[:10]]
        out["top_terms"] = [[names[i], float(coef[i])] for i in top]
    return out


def main():
    results = {}
    for v in (1, 2):
        with open(RESULTS_DIR / f"best_var{v}.json") as f:
            best = json.load(f)
        r = run_var(v, best)
        results[f"var{v}"] = r
        print(f"var{v}: expected test MSE={r['expected_test']['mse']:.4f} R2={r['expected_test']['r2']:.4f}")
        for row in r["shift_check"]:
            print("  ", row["config"], f"cv={row['cv_mse']:.4f}", f"weighted={row['test_weighted_mse']:.4f}")
        for name, t in r["other_techniques"].items():
            print("  ", name, t)

    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved", RESULTS_DIR / "analysis.json")


if __name__ == "__main__":
    main()
