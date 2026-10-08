# Polynomial Regression Assignment – BT2024267

Polynomial regression models for a two-phase geothermal power-plant expansion:

| Phase | Dataset | Inputs | Target | Final model | CV MSE | CV R² |
|---|---|---|---|---|---|---|
| 1 – Steam turbine optimisation | `var1` | x1–x6 (operational settings) | Net Power Score | degree 5, Lasso (α = 0.01) | 0.325 | 0.967 |
| 2 – Thermal reservoir mapping | `var2` | x1–x3 (3-D coordinates) | Thermal Anomaly Score | degree 10, Ridge (α = 1) | 0.249 | 0.994 |

Every model is a pure polynomial in the inputs. All monomials up to total degree *d* are generated with `PolynomialFeatures`, standardised, and fitted with OLS, Ridge or Lasso. The degree and penalty were chosen by 5-fold cross-validation repeated 3 times, over degrees 1–7 for var1 and 1–20 for var2. The full write-up is in [`report/BT2024267_report.pdf`](report/BT2024267_report.pdf).

## Deliverables

| Deliverable | Location |
|---|---|
| Report (IEEE format, PDF + LaTeX source) | `report/BT2024267_report.pdf`, `report/BT2024267_report.tex` |
| Prediction files | `predictions/BT2024267_pred_var1.csv`, `predictions/BT2024267_pred_var2.csv` |
| Training and inference code | `src/`, plus a single-cell Kaggle version in `kaggle/` |

## Repository layout

```
data/                     train/test CSVs for both problems + sample_submission.csv
src/common.py             data loading and the polynomial model pipeline
src/select_model.py       CV search over degree / estimator / alpha  -> results/cv_var*.csv, best_var*.json
src/analysis.py           train/test distribution check and other techniques -> results/analysis.json
src/predict.py            refit the chosen models on all training data and predict the test sets
src/make_report.py        generate the IEEE LaTeX report from results/ and compile it with pdflatex
kaggle/kaggle_notebook.py self-contained version for a Kaggle notebook (paste into one cell)
results/                  CV tables, selected configurations, analysis output
predictions/              final test predictions (sample-submission format: one column `y`)
report/                   the IEEE-format report (.tex source and compiled .pdf)
docs/problem_statement.txt the assignment brief
```

## Reproducing

```bash
pip install -r requirements.txt
python src/select_model.py   # degree/alpha search (about 20 min on 16 cores)
python src/analysis.py       # supporting analysis (under a minute)
python src/predict.py        # writes predictions/BT2024267_pred_var{1,2}.csv
python src/make_report.py    # rebuilds the IEEE report (needs a LaTeX distribution with IEEEtran, e.g. MiKTeX or TeX Live)
```

`results/` already contains the search output, so `predict.py` can be run directly. To predict a different test file:

```bash
python src/predict.py --var 1 --test path/to/test.csv --out my_predictions.csv
```

If the test file contains a `y` column, the script also prints the test MSE and R².

### On Kaggle

1. Attach a dataset that contains the four `BT2024267_{train,test}_var{1,2}.csv` files.
2. Paste `kaggle/kaggle_notebook.py` into a single cell and run it.
3. The prediction files are written to `/kaggle/working`.
