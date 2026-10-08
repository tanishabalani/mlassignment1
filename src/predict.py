import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from common import DATA_DIR, PRED_DIR, RESULTS_DIR, ROLL_NO, build_model, load


def run(var, test_path=None, out_path=None):
    with open(RESULTS_DIR / f"best_var{var}.json") as f:
        cfg = json.load(f)

    X, y = load("train", var)
    model = build_model(int(cfg["degree"]), cfg["method"], float(cfg["alpha"]))
    model.fit(X, y)
    p = model.predict(X)
    print(f"var{var}: degree={cfg['degree']} method={cfg['method']} alpha={cfg['alpha']}")
    print(f"  train MSE={mean_squared_error(y, p):.4f} R2={r2_score(y, p):.4f}")
    print(f"  CV    MSE={cfg['cv_mse']:.4f} R2={cfg['cv_r2']:.4f}")

    if test_path is None:
        test_path = DATA_DIR / f"{ROLL_NO}_test_var{var}.csv"
    test_path = Path(test_path)
    if not test_path.exists():
        print(f"  no test file at {test_path}, skipping")
        return

    test = pd.read_csv(test_path)
    cols = [c for c in test.columns if c.startswith("x")]
    pred = model.predict(test[cols].to_numpy())

    if out_path is None:
        out_path = PRED_DIR / f"{ROLL_NO}_pred_var{var}.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": pred}).to_csv(out_path, index=False)
    print(f"  wrote {len(pred)} predictions to {out_path}")

    if "y" in test.columns:
        print(f"  test MSE={mean_squared_error(test.y, pred):.4f} R2={r2_score(test.y, pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=int, choices=[1, 2], nargs="*", default=[1, 2])
    parser.add_argument("--test")
    parser.add_argument("--out")
    args = parser.parse_args()
    if (args.test or args.out) and len(args.var) != 1:
        parser.error("--test and --out need exactly one --var")
    for v in args.var:
        run(v, args.test, args.out)
