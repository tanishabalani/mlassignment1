import argparse
import json
import re
import shutil
import subprocess
from math import comb

import pandas as pd

from common import PRED_DIR, RESULTS_DIR, ROLL_NO, ROOT
from select_model import GRID

DEFAULT_REPO = ""
METHOD = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso"}


def f3(x):
    return f"{x:.3f}"


def term_tex(name):
    out = ""
    for part in name.split(" "):
        match = re.fullmatch(r"x(\d+)(?:\^(\d+))?", part)
        out += "x_{" + match.group(1) + "}"
        if match.group(2):
            out += "^{" + match.group(2) + "}"
    return out


def best_of(cv, method, degree):
    s = cv[(cv.method == method) & (cv.degree == degree)]
    return s.loc[s.cv_mse.idxmin()] if len(s) else None


def degree_table(cv, degrees, best, label, caption):
    lines = [r"\begin{table}[!t]", r"\renewcommand{\arraystretch}{1.12}", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}rrccc@{}}", r"\toprule",
             r"$d$ & $p(n,d)$ & OLS & Ridge ($\alpha$) & Lasso ($\alpha$) \\", r"\midrule"]
    for d in degrees:
        sub = cv[cv.degree == d]
        if sub.empty:
            continue
        cells = [str(d), str(int(sub.n_terms.iloc[0]))]
        for m in ("ols", "ridge", "lasso"):
            r = best_of(cv, m, d)
            if r is None:
                cells.append("--")
                continue
            txt = f3(r.cv_mse) + ("" if m == "ols" else f" ({r.alpha:g})")
            if d == best["degree"] and m == best["method"]:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def shift_table(an):
    rows = an["shift_check"]
    names = ["Sel.", "$d{=}4$ L", "$d{=}6$ L", "$d{=}5$ R"]
    order = [0, 1, 3, 2]
    bc = an["bound_counts"]
    lines = [r"\begin{table}[!t]", r"\renewcommand{\arraystretch}{1.12}",
             r"\caption{Phase 1 Out-of-Fold MSE by Number of Inputs on the Boundary}",
             r"\label{tab:shift}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4.2pt}",
             r"\begin{tabular}{@{}crr" + "c" * len(order) + r"@{}}", r"\toprule",
             r"$k$ & $N_{\text{train}}$ & $N_{\text{test}}$ & " + " & ".join(names) + r" \\", r"\midrule"]
    for k in range(len(bc["train"])):
        cells = [str(k), str(bc["train"][k]), str(bc["test"][k])]
        for i in order:
            v = rows[i]["per_k_mse"][k]
            cells.append("--" if v is None else f3(v))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{@{}l}{CV MSE (train mix)} & "
                 + " & ".join(f3(rows[i]["cv_mse"]) for i in order) + r" \\")
    lines.append(r"\multicolumn{3}{@{}l}{Test-weighted MSE, (\ref{eq:wmse})} & "
                 + " & ".join((r"\textbf{%s}" if i == 0 else "%s") % f3(rows[i]["test_weighted_mse"])
                              for i in order) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
              r"\parbox{0.96\columnwidth}{\scriptsize Sel.\ = selected model ($d{=}5$ Lasso, "
              r"$\alpha{=}0.01$); L = Lasso ($\alpha$ = 0.0075, 0.01); R = Ridge ($\alpha{=}30$). "
              r"No training row has $k{=}6$; for (\ref{eq:wmse}) that stratum uses the $k{=}5$ error.}",
              r"\end{table}"]
    return "\n".join(lines)


def techniques_table(cv, best, an):
    d1, d2 = int(best[1]["degree"]), int(best[2]["degree"])
    rows = [("Monomials + OLS", best_of(cv[1], "ols", d1), best_of(cv[2], "ols", d2)),
            ("Monomials + Ridge", best_of(cv[1], "ridge", d1), best_of(cv[2], "ridge", d2)),
            ("Monomials + Lasso", best_of(cv[1], "lasso", d1), best_of(cv[2], "lasso", d2))]
    lines = [r"\begin{table}[!t]", r"\renewcommand{\arraystretch}{1.12}",
             r"\caption{CV MSE of All Techniques at the Selected Degree}", r"\label{tab:tech}",
             r"\centering\footnotesize", r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             rf"Technique & Phase 1 ($d{{=}}{d1}$) & Phase 2 ($d{{=}}{d2}$) \\", r"\midrule"]
    for name, r1, r2 in rows:
        c = []
        for v, r in ((1, r1), (2, r2)):
            t = f3(r.cv_mse) if r is not None else "--"
            if r is not None and r.method == best[v]["method"]:
                t = r"\textbf{" + t + "}"
            c.append(t)
        lines.append(f"{name} & {c[0]} & {c[1]} \\\\")
    for name in ("Legendre basis + Ridge", "Legendre basis + Lasso", "Relaxed Lasso (monomials)"):
        lines.append(f"{name} & {f3(an['var1']['other_techniques'][name]['cv_mse'])} & "
                     f"{f3(an['var2']['other_techniques'][name]['cv_mse'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_env(label, caption, colspec, header, body, size=r"\footnotesize", sep="4pt"):
    return "\n".join([r"\begin{table}[!t]", r"\renewcommand{\arraystretch}{1.12}", rf"\caption{{{caption}}}",
                      rf"\label{{{label}}}", r"\centering" + size, rf"\setlength{{\tabcolsep}}{{{sep}}}",
                      rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
                     + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def describe_table(an):
    body = []
    for v, label in ((1, "Phase 1"), (2, "Phase 2")):
        a = an[f"var{v}"]
        body.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{label} (var{v})}}}} \\")
        for d in a["describe"]:
            body.append(f"${d['var'][0]}_{{{d['var'][1:]}}}$ & {d['train_mean']:+.3f} ({d['train_sd']:.3f}) & "
                        f"{d['test_mean']:+.3f} ({d['test_sd']:.3f}) & {100 * d['train_bound']:.1f} & "
                        f"{100 * d['test_bound']:.1f} \\\\")
        ty = a["train_y"]
        body.append(f"$y$ & {ty['mean']:+.3f} ({ty['std']:.3f}) & -- & -- & -- \\\\")
        if v == 1:
            body.append(r"\midrule")
    return table_env("tab:desc", "Descriptive Statistics of the Training and Test Data",
                     r"@{}lcccc@{}", r"Var. & Train mean (s.d.) & Test mean (s.d.) & \% $\pm1$ tr. & \% $\pm1$ te.",
                     body, sep="3.4pt")


def grid_table(grid):
    def al(xs):
        return ", ".join(f"{x:g}" for x in xs)

    g1, g2 = grid[1], grid[2]
    body = [
        rf"OLS, Ridge: $d$ & {min(g1['degrees'])}--{max(g1['degrees'])} & {min(g2['degrees'])}--{max(g2['degrees'])} \\",
        rf"Lasso: $d$ & {min(g1['lasso_degrees'])}--{max(g1['lasso_degrees'])} & "
        rf"{min(g2['lasso_degrees'])}--{max(g2['lasso_degrees'])} \\",
        rf"Ridge $\alpha$ & \parbox[t]{{2.2cm}}{{\raggedright {al(g1['ridge'])}}} & "
        rf"\parbox[t]{{2.2cm}}{{\raggedright {al(g2['ridge'])}}} \\",
        rf"Lasso $\alpha$ & \parbox[t]{{2.2cm}}{{\raggedright {al(g1['lasso'])}}} & "
        rf"\parbox[t]{{2.2cm}}{{\raggedright {al(g2['lasso'])}}} \\",
    ]
    return table_env("tab:grid", "Hyper-Parameter Search Space", r"@{}lll@{}",
                     r"Setting & Phase 1 & Phase 2", body)


def gap_table(an, v, sel_method, degrees, label, caption):
    tv = {(t["method"], t["degree"]): t for t in an[f"var{v}"]["train_vs_cv"]}
    sel = METHOD[sel_method]
    body = []
    for d in degrees:
        cells = [str(d)]
        for m in ("ols", sel_method):
            t = tv.get((m, d))
            cells += ["--", "--"] if t is None else [f3(t["train_mse"]), f3(t["cv_mse"])]
        body.append(" & ".join(cells) + r" \\")
    return table_env(label, caption, r"@{}rcccc@{}",
                     rf"$d$ & OLS train & OLS CV & {sel} train & {sel} CV", body)


def alpha_table(cv, best):
    body = []
    for v in (1, 2):
        b = best[v]
        s = cv[v][(cv[v].degree == b["degree"]) & (cv[v].method == b["method"])].sort_values("alpha")
        body.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{Phase {v}: $d={int(b['degree'])}$, "
                    rf"{METHOD[b['method']]}}}}} \\")
        for r in s.itertuples():
            mse = f3(r.cv_mse)
            if r.alpha == b["alpha"]:
                mse = r"\textbf{" + mse + "}"
            body.append(f"{r.alpha:g} & {mse} & {r.cv_r2:.4f} \\\\")
        if v == 1:
            body.append(r"\midrule")
    return table_env("tab:alpha", r"Sensitivity to the Penalty $\alpha$ at the Selected Degree",
                     r"@{}rcc@{}", r"$\alpha$ & CV MSE & CV $R^2$", body)


def lasso_table(an):
    a = an["var1"]
    tb = a["lasso_terms_by_degree"]
    left = [f"{d} & {tb[str(d)][0]} / {tb[str(d)][1]}" for d in range(1, len(tb) + 1)]
    right = [f"${term_tex(n)}$ & {c:+.3f}" for n, c in a["top_terms"][:10]]
    rows = max(len(left), len(right))
    left += ["&"] * (rows - len(left))
    body = [f"{l_} & {r_} \\\\" for l_, r_ in zip(left, right)]
    kept = sum(x[0] for x in tb.values())
    total = sum(x[1] for x in tb.values())
    body.append(r"\midrule")
    body.append(rf"Total & {kept} / {total} & & \\")
    return table_env("tab:lasso", "Structure of the Selected Phase 1 Lasso Polynomial", r"@{}cc|lr@{}",
                     r"Degree & Kept / total & Largest terms & $\hat\beta$ (std.)", body)


def summary_table(best, an):
    lines = [r"\begin{table}[!t]", r"\renewcommand{\arraystretch}{1.12}",
             r"\caption{Final Polynomial Models}", r"\label{tab:final}", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{3.6pt}", r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r" & Phase 1 (var1) & Phase 2 (var2) \\", r"\midrule"]
    b1, b2, a1, a2 = best[1], best[2], an["var1"], an["var2"]
    kept = sum(v[0] for v in a1["lasso_terms_by_degree"].values())
    rows = [
        ("Inputs $n$", "6", "3"),
        ("Degree $d$", str(int(b1["degree"])), str(int(b2["degree"]))),
        ("Estimator", f"Lasso, $\\alpha{{=}}{b1['alpha']:g}$", f"Ridge, $\\alpha{{=}}{b2['alpha']:g}$"),
        ("Non-zero terms", f"{kept} of {int(b1['n_terms'])}", f"{int(b2['n_terms'])} of {int(b2['n_terms'])}"),
        ("CV MSE ($\\pm$ s.e.)", f"{f3(b1['cv_mse'])} $\\pm$ {f3(b1['cv_mse_se'])}",
         f"{f3(b2['cv_mse'])} $\\pm$ {f3(b2['cv_mse_se'])}"),
        ("CV $R^2$", f"{b1['cv_r2']:.4f}", f"{b2['cv_r2']:.4f}"),
        ("Expected test MSE", f"$\\approx${f3(a1['expected_test']['mse'])}",
         f"$\\approx${f3(a2['expected_test']['mse'])}"),
        ("Expected test $R^2$", f"$\\approx${a1['expected_test']['r2']:.3f}",
         f"$\\approx${a2['expected_test']['r2']:.3f}"),
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


TEMPLATE = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{cite}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\begin{document}

\title{Degree Selection for Polynomial Regression in a Geothermal Power Plant Expansion: Turbine Power Prediction and Thermal Reservoir Mapping}

\author{\IEEEauthorblockN{<<AUTHOR>>}
\IEEEauthorblockA{Polynomial Regression Assignment\\
Datasets: \texttt{<<ROLL>>\_\{train,test\}\_var1}, \texttt{<<ROLL>>\_\{train,test\}\_var2}\\
GitHub repository: <<REPO>>}}

\maketitle

\begin{abstract}
Two personalised regression problems from a geothermal plant expansion are solved using only polynomial regression: predicting a steam turbine's Net Power Score from six operational settings (Phase~1) and a subsurface Thermal Anomaly Score from three spatial coordinates (Phase~2). All monomials up to a total degree $d$ are generated and fitted by ordinary least squares, ridge or lasso. The degree and penalty are chosen by 5-fold cross-validation repeated three times over <<NCFG1>> and <<NCFG2>> candidate configurations. Phase~1 is best modelled by a sparse degree-<<D1>> polynomial (lasso, <<KEPT>> of <<P1>> terms retained; CV MSE <<CV1>>, $R^2$ <<R21>>). Phase~2 is best modelled by a dense degree-<<D2>> polynomial (ridge; CV MSE <<CV2>>, $R^2$ <<R22>>). Because the Phase~1 test inputs lie on the boundary of the input domain far more often than the training inputs, the CV errors are additionally re-weighted to the test distribution. This confirms the model choice and gives an expected test MSE of about <<EXP1>>.
\end{abstract}

\begin{IEEEkeywords}
polynomial regression, model selection, cross-validation, ridge regression, lasso, bias--variance trade-off
\end{IEEEkeywords}

\section{Introduction}
As lead renewable energy engineer for a multi-stage geothermal plant expansion, two predictive models are required. In \emph{Phase~1} the surface plant is optimised: the Net Power Score $y$ of a multi-stage steam turbine must be predicted from six operational parameters. In \emph{Phase~2} drilling sites are selected: a Thermal Anomaly Score $y$ must be predicted at untested 3-D coordinates from a random scatter of core samples. The assignment restricts the model class to polynomial regression and states that the Phase~1 response is a polynomial of moderate degree (at most 10) and the Phase~2 response one of high degree (at most 20). Predictions are evaluated on hidden test targets by mean squared error (MSE) and the coefficient of determination ($R^2$).

The central decision is therefore the polynomial degree. Too low a degree under-fits (high bias). Too high a degree introduces more coefficients than 1000 samples can determine, so the model fits noise (high variance) \cite{hastie2009}. This report documents the approach (Section~\ref{sec:method}), the degree chosen for each problem and its rationale (Section~\ref{sec:results}), the other techniques evaluated (Section~\ref{sec:other}), a discussion of sensitivity and limitations (Section~\ref{sec:disc}) and the final models (Section~\ref{sec:final}).

\section{Problem Statement and Data}\label{sec:data}
\begin{table}[!t]
\renewcommand{\arraystretch}{1.12}
\caption{Input and Target Variables}
\label{tab:vars}
\centering\footnotesize
\begin{tabular}{@{}cll@{}}
\toprule
Phase & Variable & Description \\
\midrule
1 & $x_1$ & High-pressure steam valve adjustment \\
1 & $x_2$ & Condenser coolant flow rate adjustment \\
1 & $x_3$ & Re-injection pump hydraulic pressure \\
1 & $x_4$ & Turbine blade pitch angle \\
1 & $x_5$ & Non-condensable gas exhaust valve rate \\
1 & $x_6$ & Steam inlet pressure adjustment \\
1 & $y$ & Net Power Score \\
\midrule
2 & $x_1$ & East--West coordinate offset \\
2 & $x_2$ & North--South coordinate offset \\
2 & $x_3$ & Vertical depth offset from base camp \\
2 & $y$ & Thermal Anomaly Score \\
\bottomrule
\end{tabular}
\end{table}

The variables are listed in Table~\ref{tab:vars}. Each problem has 1000 training rows with targets and 1000 test rows without targets. There are no missing values. All inputs lie in $[-1,1]$, and many values sit \emph{exactly} on the bounds $\pm1$, which indicates that the inputs were clipped to the box. In Phase~1, <<FB1TR>>\% of training input values lie on the boundary, compared with <<FB1TE>>\% of test values. The Phase~1 test set therefore contains many more points near the corners of the 6-D domain, including <<K6>> rows with all six settings at a limit, a configuration absent from the training data. Phase~2 is balanced (<<FB2TR>>\% vs.\ <<FB2TE>>\%). The training targets span $[<<Y1MIN>>, <<Y1MAX>>]$ (standard deviation <<Y1SD>>) in Phase~1 and $[<<Y2MIN>>, <<Y2MAX>>]$ (standard deviation <<Y2SD>>) in Phase~2.

<<TAB_DESC>>

Table~\ref{tab:desc} gives per-variable statistics. All inputs are approximately centred (means within $\pm0.04$ of zero) with similar spread, so no input dominates the polynomial expansion. Additional scaling before the expansion is therefore unnecessary, and the standardisation in Section~\ref{sec:method} is applied only to the generated monomials. The boundary shift in Phase~1 is not caused by a single variable: every input moves from roughly 30\% of values at $\pm1$ in training to roughly 50\% in testing. This affects the number of saturated settings per row, $k$, which is why Section~\ref{sec:method}-D stratifies by $k$. In Phase~2 the train and test proportions agree to within three percentage points for every coordinate.

\section{Methodology}\label{sec:method}
\subsection{Polynomial Feature Expansion}
For an input $\mathbf{x}\in[-1,1]^n$, a polynomial of degree $d$ is
\begin{equation}
f(\mathbf{x}) = \beta_0 + \sum_{\mathbf{a}\in\mathcal{A}_d} \beta_{\mathbf{a}} \prod_{j=1}^{n} x_j^{a_j},
\label{eq:poly}
\end{equation}
where $\mathcal{A}_d=\{\mathbf{a}\in\mathbb{N}^n : 1\le \sum_j a_j\le d\}$, so that, as the assignment specifies, the powers in every term sum to at most $d$. The number of non-constant terms is
\begin{equation}
p(n,d)=\binom{n+d}{d}-1 ,
\label{eq:count}
\end{equation}
which grows rapidly. For $n=6$, $p=<<P6_3>>, <<P6_4>>, <<P6_5>>, <<P6_6>>$ at $d=3,\dots,6$, and a full degree-10 basis would have $p=<<P6_10>>$. For $n=3$, $p=<<P3_8>>, <<P3_10>>, <<P3_20>>$ at $d=8, 10, 20$. The monomials are generated with scikit-learn's \texttt{PolynomialFeatures} \cite{pedregosa2011}. Each column $\phi_m$ of the resulting design matrix $\boldsymbol{\Phi}\in\mathbb{R}^{N\times p}$ is standardised as
\begin{equation}
\tilde\phi_m(\mathbf{x}) = \frac{\phi_m(\mathbf{x})-\mu_m}{\sigma_m},
\label{eq:std}
\end{equation}
where $\mu_m$ and $\sigma_m$ are the mean and standard deviation of the column on the training fold. Standardisation is an affine rescaling of each column, so substituting (\ref{eq:std}) back gives coefficients $\beta_m/\sigma_m$ and an adjusted intercept. The fitted model is therefore still exactly a degree-$d$ polynomial in the raw inputs. Standardisation matters for two reasons. On $[-1,1]$, high powers such as $x^{10}$ have far smaller variance than linear terms, so without it a penalty would act unevenly across degrees. It also reduces the condition number of $\boldsymbol{\Phi}^\top\boldsymbol{\Phi}$.

\subsection{Estimators}
Three estimators were compared at every degree. Ordinary least squares (OLS),
\begin{equation}
\hat{\boldsymbol\beta}_{\text{OLS}}=\arg\min_{\boldsymbol\beta}\ \lVert \mathbf{y}-\boldsymbol{\Phi}\boldsymbol\beta\rVert_2^2 ,
\end{equation}
was used only when $p$ is below 80\% of the training-fold size. Ridge regression \cite{hoerl1970},
\begin{equation}
\hat{\boldsymbol\beta}_{\text{R}}=\arg\min_{\boldsymbol\beta}\ \lVert \mathbf{y}-\boldsymbol{\Phi}\boldsymbol\beta\rVert_2^2+\alpha\lVert\boldsymbol\beta\rVert_2^2 ,
\end{equation}
shrinks all coefficients, which stabilises the highly collinear monomials (e.g., $x^4$ and $x^6$). The lasso \cite{tibshirani1996},
\begin{equation}
\hat{\boldsymbol\beta}_{\text{L}}=\arg\min_{\boldsymbol\beta}\ \tfrac{1}{2N}\lVert \mathbf{y}-\boldsymbol{\Phi}\boldsymbol\beta\rVert_2^2+\alpha\lVert\boldsymbol\beta\rVert_1 ,
\end{equation}
sets many coefficients exactly to zero and so performs term selection: a high degree can be used while estimating only the terms that the data supports.

\subsection{Degree and Penalty Selection}
Each candidate $(d,\text{estimator},\alpha)$ was scored by 5-fold cross-validation repeated three times \cite{kohavi1995}, i.e., 15 train/validation splits. The same splits were used for every candidate, so all comparisons are paired. The grid covered $d=1,\dots,7$ for Phase~1 (<<NCFG1>> configurations) and $d=1,\dots,20$ for Phase~2 (<<NCFG2>> configurations), with $\alpha$ on a logarithmic grid. The selection criterion is the mean validation MSE,
\begin{equation}
\text{MSE}=\frac{1}{N}\sum_{i=1}^{N}\bigl(y_i-\hat y_i\bigr)^2 ,
\end{equation}
reported together with
\begin{equation}
R^2 = 1-\frac{\text{MSE}}{\operatorname{Var}(y)} .
\end{equation}
The configuration with the lowest mean CV MSE is refitted on all 1000 training rows to produce the test predictions.

\subsection{Re-weighting to the Test Input Distribution}
Because the Phase~1 test inputs lie on the boundary more often than the training inputs (Section~\ref{sec:data}), plain CV may be optimistic. Let $k$ be the number of a row's inputs equal to $\pm1$, let $\bar e_k$ be the mean out-of-fold squared error of training rows in stratum $k$, and let $\pi_k^{\text{test}}$ be the fraction of test rows in that stratum. The test-weighted error
\begin{equation}
\text{MSE}_{w}=\sum_{k=0}^{n}\pi_k^{\text{test}}\,\bar e_k
\label{eq:wmse}
\end{equation}
is a stratified importance-weighted estimate under covariate shift \cite{shimodaira2000}. It is used to re-check the model choice and to estimate the test error. An expected test $R^2$ is obtained by approximating $\operatorname{Var}(y_{\text{test}})$ as the variance of the test predictions plus $\text{MSE}_w$.

\subsection{Implementation and Procedure}
<<TAB_GRID>>
The search space is listed in Table~\ref{tab:grid}. Ridge is solved in closed form and the lasso by cyclic coordinate descent (up to $5\times10^4$ iterations, tolerance $10^{-5}$) \cite{pedregosa2011}. The lasso is searched only from $d=3$ for Phase~1 and $d=5$ for Phase~2, because lower degrees are clearly under-fitted. The complete procedure for each problem is:
\begin{enumerate}
\item Load the 1000 training rows and generate the 15 repeated 5-fold splits (random seed 42).
\item For every configuration in Table~\ref{tab:grid}, fit the pipeline \emph{polynomial expansion} $\rightarrow$ \emph{standardisation} $\rightarrow$ \emph{estimator} on each training fold and record the validation MSE. Standardisation statistics are computed on the training fold only, so no information leaks into validation.
\item Average over the 15 splits and select the configuration with the lowest mean CV MSE.
\item Verify the choice with the test-weighted error (\ref{eq:wmse}) and compare it with alternative polynomial techniques (Section~\ref{sec:other}).
\item Refit the selected pipeline on all 1000 rows and predict the 1000 test rows.
\end{enumerate}

\section{Results and Degree Selection}\label{sec:results}
\subsection{Phase 1: Steam Turbine Net Power Score}
<<TAB_DEG1>>

Table~\ref{tab:deg1} lists the best CV MSE per degree for each estimator. With OLS the error falls steeply up to $d=<<OLSD1>>$ (MSE <<OLSM1>>). At $d=5$ ($p=<<P6_5>>$) the error rises to <<OLS51>> as variance dominates; lower degrees under-fit. Regularisation moves the optimum to a higher degree. The lasso is best overall at \textbf{$d=<<D1>>$, $\alpha=<<A1>>$} with CV MSE <<CV1>>~$\pm$~<<SE1>> ($R^2=<<R21>>$), against <<L4>> at $d=4$ and <<L6>> at $d=6$. Degree <<D1>> is therefore the minimum of the validation curve. Beyond it the error rises slowly (<<L7>> at $d=7$). A full degree-10 basis ($p=<<P6_10>>$) cannot be identified from 1000 samples.

<<TAB_GAP1>>

Table~\ref{tab:gap1} compares the training error with the CV error for the best configuration of each degree, which shows the bias--variance trade-off directly. Up to $d=3$ the training and CV errors are both large and close together: the model under-fits. For OLS the gap then widens sharply. At $d=5$ the training MSE falls to <<GAPOLS5TR>> while the CV MSE rises to <<OLS51>>, a ten-fold difference that is the signature of over-fitting. The lasso keeps the gap small at every degree. Its training error barely improves beyond $d=5$ (<<GAPL5TR>> at $d=5$, <<GAPL7TR>> at $d=7$) while its CV error worsens, so the extra terms of degrees 6 and 7 capture noise rather than signal.

<<TAB_LASSO>>

The lasso clearly outperforms ridge at the same degree (<<CV1>> vs.\ <<R51>>), which indicates that the true response is \emph{sparse}: it requires some degree-5 interactions but far fewer than all of them. Table~\ref{tab:lasso} shows the structure of the selected polynomial. Of the <<P1>> candidate monomials the lasso retains <<KEPT>>, and only <<KEPTLOW>> of the <<TOTLOW>> terms of degree $\le3$ survive. The response is dominated by pairwise interactions such as $x_1x_5$ and $x_3x_6$, by mixed terms such as $x_2^3x_3$, and by a set of degree-4 and degree-5 interactions. This structure explains why a full dense degree-5 basis (OLS or ridge) over-fits: most of its 461 coefficients estimate zero effects.

<<TAB_SHIFT>>

Table~\ref{tab:shift} applies the re-weighting of (\ref{eq:wmse}). Errors increase towards the boundary for every model, and the test set is concentrated at $k=3$--$5$, so the expected test MSE of the selected model is <<EXP1>> rather than <<CV1>>. The selected model remains the best candidate after re-weighting, and its margin over the degree-4 lasso and the ridge model widens. The test predictions also vary more than the training targets (standard deviation <<PSD1>> vs.\ <<Y1SD>>), so the expected test $R^2$ remains about <<EXPR21>>.

\subsection{Phase 2: Thermal Anomaly Score}
<<TAB_DEG2>>

Table~\ref{tab:deg2} shows that the thermal field is genuinely high-order. OLS improves up to $d=<<OLSD2>>$ (MSE <<OLSM2>>) and then degrades rapidly (<<OLS102>> at $d=10$, <<OLS122>> at $d=12$). With ridge, the error keeps falling to its minimum at \textbf{$d=<<D2>>$, $\alpha=<<A2>>$}: CV MSE <<CV2>>~$\pm$~<<SE2>> ($R^2=<<R22>>$), compared with <<R92>> at $d=9$ and <<R112>> at $d=11$. <<ONESE2>> Raising the degree towards the stated maximum of 20 only adds variance (<<R202>> at $d=20$). Unlike Phase~1, the lasso gives no gain (<<LAS102>> at $d=10$): the response is dense in the monomial basis, so shrinking all <<P2>> coefficients works better than selecting a subset. The Phase~2 test inputs follow the training distribution, and the test-weighted MSE (<<EXP2>>) agrees with plain CV. The selected model also stays ahead of degree-8 OLS (<<W8O>>), degree-8 ridge (<<W8R>>) and degree-12 ridge (<<W12R>>) under re-weighting. Its stratum counts are almost identical in training and testing (<<K2TR>> vs.\ <<K2TE>> rows for $k=0,\dots,3$).

<<TAB_GAP2>>

Table~\ref{tab:gap2} shows the same bias--variance pattern in three dimensions. The OLS training error keeps decreasing with degree, reaching <<GAPOLS13TR>> at $d=13$, while its CV error explodes to <<OLS132>>. Ridge holds the training error near <<GAPR10TR>> and the CV error near its minimum. For ridge, raising $d$ beyond 10 lowers the training error only marginally while the CV error creeps up, so $d=10$ is the point of best generalisation.

\section{Other Techniques Evaluated}\label{sec:other}
<<TAB_TECH>>
Table~\ref{tab:tech} compares further polynomial-regression variants at the selected degrees, scored on the same 15 splits with the best $\alpha$ for each.

\emph{Orthogonal basis.} Products of Legendre polynomials $P_{a_1}(x_1)\cdots P_{a_n}(x_n)$ \cite{xiu2002} span exactly the same space as the monomials in (\ref{eq:poly}), but are orthogonal on $[-1,1]$. Penalising coefficients in this basis was worse in both phases (<<LEGL1>> and <<LEGR2>>). The targets have a simpler structure in the monomial basis, which is the basis the assignment uses to define the degree.

\emph{Relaxed lasso.} Here the lasso selects the terms and an unpenalised OLS refit removes shrinkage bias \cite{meinshausen2007}. It matched the lasso in Phase~1 (<<RLX1>>) without improving on it, and was worse in Phase~2 (<<RLX2>>), so the simpler estimators were retained.


\section{Discussion}\label{sec:disc}
\subsection{Sensitivity to the Penalty}
<<TAB_ALPHA>>
Table~\ref{tab:alpha} shows how the CV error changes with $\alpha$ at the selected degrees. Both optima are broad. In Phase~1 every $\alpha$ between <<AL1LO>> and <<AL1HI>> gives a CV MSE within <<AL1SPREAD>> of the best. Too small an $\alpha$ lets noise terms into the model, and too large an $\alpha$ removes genuine degree-5 interactions. In Phase~2 the penalty matters more. The CV MSE falls steadily from <<AL2LO>> at $\alpha=<<AL2LOA>>$ to <<CV2>> at $\alpha=<<A2>>$, because a tiny penalty leaves the degree-10 fit almost as unstable as OLS. It then rises again to <<AL2HI>> at $\alpha=<<AL2HIA>>$, where shrinkage starts to bias the high-order terms. In both phases the selected $\alpha$ lies inside the searched grid rather than at its edge, so the optimum is bracketed and the grid did not need extending.

\subsection{Sparse Versus Dense Polynomials}
The two problems need different estimators even though both are polynomial regressions. The Phase~1 response depends on six variables, but only about a quarter of its possible interaction terms are active. The lasso can find them, so the model reaches degree 5 with only <<KEPT>> estimated coefficients. Ridge, by contrast, must spread its shrinkage across all 461 terms, and OLS cannot shrink at all. The Phase~2 thermal field depends on only three coordinates but is smooth and genuinely high-order, and the lasso finds no sparse subset of its 285 terms that predicts better. Ridge suits it because it keeps every term and only damps the coefficients that the data cannot pin down.

\subsection{Achievable Accuracy}
The best CV MSE approximates the error of the true polynomial plus noise. For the selected models the training MSE (<<GAPL5TR>> and <<GAPR10TR>>) lies below the CV MSE (<<CV1>> and <<CV2>>), as expected for flexible models. The small gap shows that little of the fit is memorisation. The remaining CV error probably consists mostly of noise that no model could remove: in Phase~2 several regularised configurations from $d=8$ to $d=14$ reach almost the same error floor of about 0.25--0.27.

\subsection{Limitations}
The test-weighted estimate (\ref{eq:wmse}) assumes that, within each stratum $k$, the test rows resemble the training rows. The <<K6>> Phase~1 test rows with all six inputs at a limit have no training counterpart, so their error was approximated by the $k=5$ stratum and could be larger. Cross-validation also selects the configuration with the lowest \emph{estimated} error. With 15 splits, differences smaller than one standard error (about <<SE1>> in Phase~1 and <<SE2>> in Phase~2) cannot be resolved reliably. For this reason the analysis checks that the curve is flat around each optimum.

\section{Final Models and Test Predictions}\label{sec:final}
<<TAB_FINAL>>
Table~\ref{tab:final} summarises the selected models. Both were refitted on all 1000 training rows and applied to the 1000 test rows of each problem. The predictions are stored in \texttt{<<ROLL>>\_pred\_var1.csv} and \texttt{<<ROLL>>\_pred\_var2.csv} in the sample-submission format (a single column $y$, in test-row order). The Phase~1 predictions have mean <<PM1>> and range $[<<PMIN1>>, <<PMAX1>>]$; the Phase~2 predictions have mean <<PM2>> and range $[<<PMIN2>>, <<PMAX2>>]$. Both are consistent with the training targets.

\section{Conclusion}
The degree of each polynomial was selected where repeated cross-validation error is lowest, between under-fitting at low degree and the variance explosion of unregularised high-degree fits. The steam-turbine response is a sparse degree-<<D1>> polynomial, recovered by the lasso with <<KEPT>> active terms (CV $R^2=<<R21>>$). The thermal reservoir field is a dense degree-<<D2>> polynomial, fitted by ridge regression (CV $R^2=<<R22>>$). Re-weighting the validation errors to the test input distribution confirmed both choices. It also showed that the boundary-heavy Phase~1 test set should be expected to give a somewhat higher MSE (about <<EXP1>>) than plain CV suggests. All code for training, model selection, inference and report generation is provided in the accompanying GitHub repository, together with a script that regenerates every number in this report.

\begin{thebibliography}{8}
\bibitem{hastie2009} T.~Hastie, R.~Tibshirani, and J.~Friedman, \emph{The Elements of Statistical Learning}, 2nd~ed. New York, NY, USA: Springer, 2009.
\bibitem{pedregosa2011} F.~Pedregosa \emph{et al.}, ``Scikit-learn: Machine learning in Python,'' \emph{J. Mach. Learn. Res.}, vol.~12, pp.~2825--2830, 2011.
\bibitem{hoerl1970} A.~E. Hoerl and R.~W. Kennard, ``Ridge regression: Biased estimation for nonorthogonal problems,'' \emph{Technometrics}, vol.~12, no.~1, pp.~55--67, 1970.
\bibitem{tibshirani1996} R.~Tibshirani, ``Regression shrinkage and selection via the lasso,'' \emph{J. Roy. Statist. Soc. B}, vol.~58, no.~1, pp.~267--288, 1996.
\bibitem{kohavi1995} R.~Kohavi, ``A study of cross-validation and bootstrap for accuracy estimation and model selection,'' in \emph{Proc. 14th Int. Joint Conf. Artif. Intell.}, 1995, pp.~1137--1143.
\bibitem{shimodaira2000} H.~Shimodaira, ``Improving predictive inference under covariate shift by weighting the log-likelihood function,'' \emph{J. Statist. Plann. Inference}, vol.~90, no.~2, pp.~227--244, 2000.
\bibitem{xiu2002} D.~Xiu and G.~E. Karniadakis, ``The Wiener--Askey polynomial chaos for stochastic differential equations,'' \emph{SIAM J. Sci. Comput.}, vol.~24, no.~2, pp.~619--644, 2002.
\bibitem{meinshausen2007} N.~Meinshausen, ``Relaxed lasso,'' \emph{Comput. Statist. Data Anal.}, vol.~52, no.~1, pp.~374--393, 2007.
\end{thebibliography}

\end{document}
"""


def build(repo_url, author):
    best = {v: json.load(open(RESULTS_DIR / f"best_var{v}.json")) for v in (1, 2)}
    cv = {v: pd.read_csv(RESULTS_DIR / f"cv_var{v}.csv") for v in (1, 2)}
    an = json.load(open(RESULTS_DIR / "analysis.json"))
    a1, a2 = an["var1"], an["var2"]
    preds = {v: pd.read_csv(PRED_DIR / f"{ROLL_NO}_pred_var{v}.csv").y for v in (1, 2)}
    b1, b2 = best[1], best[2]
    d1, d2 = int(b1["degree"]), int(b2["degree"])

    ols1 = cv[1][cv[1].method == "ols"].sort_values("cv_mse").iloc[0]
    ols2 = cv[2][cv[2].method == "ols"].sort_values("cv_mse").iloc[0]
    tb = a1["lasso_terms_by_degree"]
    close = cv[2][cv[2].cv_mse <= b2["cv_mse"] + b2["cv_mse_se"]]
    others = sorted(set(int(x) for x in close.degree) - {d2})
    onese = (f"Degrees {', '.join(map(str, others))} lie within one standard error of the minimum, "
             f"so the validation curve is flat around $d={d2}$ and the choice is robust."
             if others else "")
    sc2 = {tuple(r["config"]): r["test_weighted_mse"] for r in a2["shift_check"]}

    def m(table, method, degree):
        return f3(best_of(table, method, degree).cv_mse)

    gap ={v: {(t["method"], t["degree"]): t["train_mse"] for t in an[f"var{v}"]["train_vs_cv"]} for v in (1, 2)}
    s1 = cv[1][(cv[1].degree == d1) & (cv[1].method == b1["method"])]
    near = s1[s1.cv_mse <= b1["cv_mse"] + 0.02]
    s2 = cv[2][(cv[2].degree == d2) & (cv[2].method == b2["method"])].sort_values("alpha")
    if repo_url:
        repo = r"\url{" + repo_url + "}"
    else:
        repo = r"\rule[-0.2ex]{5.5cm}{0.4pt}"
    if author:
        author_line = f"{author} (Roll No. {ROLL_NO})"
    else:
        author_line = f"Roll No. {ROLL_NO}"

    vals = {
        "AUTHOR": author_line,
        "ROLL": ROLL_NO,
        "REPO": repo,
        "TAB_DESC": describe_table(an), "TAB_GRID": grid_table(GRID),
        "TAB_GAP1": gap_table(an, 1, b1["method"], range(1, 8), "tab:gap1",
                              "Phase 1: Training vs.\\ CV MSE (Best $\\alpha$ per Degree)"),
        "TAB_GAP2": gap_table(an, 2, b2["method"], [6, 8, 9, 10, 11, 12, 13, 14, 16, 20], "tab:gap2",
                              "Phase 2: Training vs.\\ CV MSE (Best $\\alpha$ per Degree)"),
        "TAB_LASSO": lasso_table(an), "TAB_ALPHA": alpha_table(cv, best),
        "GAPOLS5TR": f3(gap[1][("ols", 5)]), "GAPL5TR": f3(gap[1][("lasso", 5)]),
        "GAPL7TR": f3(gap[1][("lasso", 7)]), "GAPOLS13TR": f3(gap[2][("ols", 13)]),
        "GAPR10TR": f3(gap[2][("ridge", 10)]), "OLS132": m(cv[2], "ols", 13),
        "KEPTLOW": str(sum(tb[str(d)][0] for d in (1, 2, 3))), "TOTLOW": str(sum(tb[str(d)][1] for d in (1, 2, 3))),
        "K2TR": ", ".join(map(str, a2["bound_counts"]["train"])),
        "K2TE": ", ".join(map(str, a2["bound_counts"]["test"])),
        "AL1LO": f"{near.alpha.min():g}", "AL1HI": f"{near.alpha.max():g}", "AL1SPREAD": "0.02",
        "AL2LO": f3(s2.iloc[0].cv_mse), "AL2LOA": f"{s2.iloc[0].alpha:g}",
        "AL2HI": f3(s2.iloc[-1].cv_mse), "AL2HIA": f"{s2.iloc[-1].alpha:g}",
        "NCFG1": str(len(cv[1])), "NCFG2": str(len(cv[2])),
        "D1": str(d1), "D2": str(d2), "A1": f"{b1['alpha']:g}", "A2": f"{b2['alpha']:g}",
        "P1": str(int(b1["n_terms"])), "P2": str(int(b2["n_terms"])),
        "CV1": f3(b1["cv_mse"]), "CV2": f3(b2["cv_mse"]), "SE1": f3(b1["cv_mse_se"]), "SE2": f3(b2["cv_mse_se"]),
        "R21": f"{b1['cv_r2']:.3f}", "R22": f"{b2['cv_r2']:.3f}",
        "KEPT": str(sum(v[0] for v in tb.values())),
        "KEPTBYDEG": ", ".join(str(tb[str(d)][0]) for d in range(1, d1 + 1)),
        "TOPTERMS": ", ".join(f"${term_tex(n)}$" for n, _ in a1["top_terms"][:6]),
        "EXP1": f3(a1["expected_test"]["mse"]), "EXP2": f3(a2["expected_test"]["mse"]),
        "EXPR21": f"{a1['expected_test']['r2']:.3f}",
        "FB1TR": f"{100 * a1['frac_values_at_bound']['train']:.0f}",
        "FB1TE": f"{100 * a1['frac_values_at_bound']['test']:.0f}",
        "FB2TR": f"{100 * a2['frac_values_at_bound']['train']:.0f}",
        "FB2TE": f"{100 * a2['frac_values_at_bound']['test']:.0f}",
        "K6": str(a1["bound_counts"]["test"][6]),
        "Y1MIN": f"{a1['train_y']['min']:.1f}", "Y1MAX": f"{a1['train_y']['max']:.1f}",
        "Y1SD": f"{a1['train_y']['std']:.2f}", "Y2MIN": f"{a2['train_y']['min']:.1f}",
        "Y2MAX": f"{a2['train_y']['max']:.1f}", "Y2SD": f"{a2['train_y']['std']:.2f}",
        "P6_3": str(comb(9, 3) - 1), "P6_4": str(comb(10, 4) - 1), "P6_5": str(comb(11, 5) - 1),
        "P6_6": str(comb(12, 6) - 1), "P6_10": str(comb(16, 10) - 1),
        "P3_8": str(comb(11, 8) - 1), "P3_10": str(comb(13, 10) - 1), "P3_20": str(comb(23, 20) - 1),
        "OLSD1": str(int(ols1.degree)), "OLSM1": f3(ols1.cv_mse), "OLS51": m(cv[1], "ols", 5),
        "L4": m(cv[1], "lasso", 4), "L6": m(cv[1], "lasso", 6), "L7": m(cv[1], "lasso", 7),
        "R51": m(cv[1], "ridge", 5),
        "OLSD2": str(int(ols2.degree)), "OLSM2": f3(ols2.cv_mse), "OLS102": m(cv[2], "ols", 10),
        "OLS122": m(cv[2], "ols", 12), "R92": m(cv[2], "ridge", 9), "R112": m(cv[2], "ridge", 11),
        "R202": m(cv[2], "ridge", 20), "LAS102": m(cv[2], "lasso", 10), "ONESE2": onese,
        "W8O": f3(sc2[(8, "ols", 0.0)]), "W8R": f3(sc2[(8, "ridge", 0.1)]), "W12R": f3(sc2[(12, "ridge", 1.0)]),
        "PSD1": f"{a1['test_pred']['std']:.2f}",
        "LEGL1": f"{f3(a1['other_techniques']['Legendre basis + Lasso']['cv_mse'])} vs.\\ {f3(b1['cv_mse'])}",
        "LEGR2": f"{f3(a2['other_techniques']['Legendre basis + Ridge']['cv_mse'])} vs.\\ {f3(b2['cv_mse'])}",
        "RLX1": f3(a1["other_techniques"]["Relaxed Lasso (monomials)"]["cv_mse"]),
        "RLX2": f3(a2["other_techniques"]["Relaxed Lasso (monomials)"]["cv_mse"]),
        "PM1": f"{preds[1].mean():.2f}", "PMIN1": f"{preds[1].min():.1f}", "PMAX1": f"{preds[1].max():.1f}",
        "PM2": f"{preds[2].mean():.2f}", "PMIN2": f"{preds[2].min():.1f}", "PMAX2": f"{preds[2].max():.1f}",
        "TAB_DEG1": degree_table(cv[1], range(1, 8), b1, "tab:deg1",
                                 "Phase 1: Best CV MSE per Degree ($n=6$)"),
        "TAB_DEG2": degree_table(cv[2], [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20], b2, "tab:deg2",
                                 "Phase 2: Best CV MSE per Degree ($n=3$)"),
        "TAB_SHIFT": shift_table(a1),
        "TAB_TECH": techniques_table(cv, best, an),
        "TAB_FINAL": summary_table(best, an),
    }
    tex = TEMPLATE
    for k, v in vals.items():
        tex = tex.replace(f"<<{k}>>", v)
    missing = re.findall(r"<<\w+>>", tex)
    if missing:
        raise ValueError(f"unfilled placeholders: {missing}")

    out_dir = ROOT / "report"
    out_dir.mkdir(exist_ok=True)
    tex_path = out_dir / f"{ROLL_NO}_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print("wrote", tex_path)

    if shutil.which("pdflatex") is None:
        print("pdflatex not found - compile the .tex with any LaTeX distribution (IEEEtran class).")
        return
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                           cwd=out_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise RuntimeError("pdflatex failed")
    for ext in (".aux", ".log", ".out"):
        (out_dir / f"{ROLL_NO}_report{ext}").unlink(missing_ok=True)
    print("wrote", out_dir / f"{ROLL_NO}_report.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-url", default=DEFAULT_REPO, help="GitHub URL to print in the report (blank by default)")
    p.add_argument("--author", default="", help="name shown in the author block (roll number is always shown)")
    a = p.parse_args()
    build(a.repo_url, a.author)
