import numpy as np
import warnings
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t, beta, gamma, weibull_min, lognorm

warnings.filterwarnings("ignore")

# =============================================================================
# 0. ADAPTIVE STAG MODEL (Stable version: improved numerical stability)
# =============================================================================

def convert_ab(a_raw, b_raw):
    """Map raw parameters to boundaries [a, b] using sigmoid for bounded values."""
    a = -0.3 + 0.6 / (1 + np.exp(-a_raw))   # a ∈ [0, 0.3]
    b = 0.7 + 0.6 / (1 + np.exp(-b_raw))    # b ∈ [0.7, 1.3]
    return a, b


def stag_pdf(x, mu, sigma_L, sigma_R, a_raw, b_raw):
    """
    STAG probability density function.
    Piecewise Gaussian with different std on left/right, normalized via numerical integration.
    """
    a, b = convert_ab(a_raw, b_raw)
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)
    mask = (x >= a) & (x <= b)
    x_valid = x[mask]

    if sigma_L <= 0 or sigma_R <= 0 or a >= b:
        return pdf

    # Compute unnormalized PDF on support
    x_grid = np.linspace(a, b, 1000)
    pdf_grid = np.where(
        x_grid <= mu,
        norm.pdf(x_grid, mu, sigma_L),
        norm.pdf(x_grid, mu, sigma_R)
    )
    norm_const = np.trapz(pdf_grid, x_grid)
    if norm_const < 1e-15:
        return pdf

    # Normalized PDF
    pdf[mask] = np.where(
        x_valid <= mu,
        norm.pdf(x_valid, mu, sigma_L),
        norm.pdf(x_valid, mu, sigma_R)
    ) / norm_const
    return pdf


def stag_cdf(x, mu, sigma_L, sigma_R, a_raw, b_raw):
    """
    STAG cumulative distribution function.
    Computed via numerical integration of PDF for consistency.
    """
    a, b = convert_ab(a_raw, b_raw)
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, a, b)

    if sigma_L <= 0 or sigma_R <= 0 or a >= b:
        return np.where(x < a, 0.0, 1.0).astype(float)

    # Build fine grid on support
    x_grid = np.linspace(a, b, 2000)
    pdf_grid = np.where(
        x_grid <= mu,
        norm.pdf(x_grid, mu, sigma_L),
        norm.pdf(x_grid, mu, sigma_R)
    )
    norm_const = np.trapz(pdf_grid, x_grid)
    if norm_const < 1e-15:
        return np.where(x < a, 0.0, 1.0).astype(float)

    pdf_grid /= norm_const
    dx = x_grid[1] - x_grid[0]
    cdf_grid = np.cumsum(pdf_grid) * dx
    cdf_grid = np.clip(cdf_grid, 0, 1)

    result = np.interp(x_clipped, x_grid, cdf_grid, left=0.0, right=1.0)
    result = np.where(x <= a, 0.0, result)
    result = np.where(x >= b, 1.0, result)
    return result


def nll_adaptive_stag(params, y):
    """Negative log-likelihood for optimization."""
    mu, sL, sR, a_raw, b_raw = params
    pdf = stag_pdf(y, mu, sL, sR, a_raw, b_raw)
    pdf = np.clip(pdf, 1e-15, 1e15)
    return -np.sum(np.log(pdf))


def fit_stag(y):
    """Fit STAG parameters (multi-start + strict success criteria)."""
    y = np.clip(y, 1e-5, 1 - 1e-5)
    m = np.median(y)
    s = np.std(y) if np.std(y) > 0 else 0.1
    
    # Multiple initial points
    starts = [
        [m, s, s * 1.2, 0.0, 0.0],
        [m, s * 0.8, s * 1.2, 0.0, 0.0],
        [m * 1.1, s, s * 0.9, 0.0, 0.0],
        [m * 0.9, s * 1.1, s * 1.3, 0.0, 0.0],
    ]
    
    bounds = [(0.01, 0.99), (0.01, 1.0), (0.01, 1.0), (-5, 5), (-5, 5)]
    best_result = None
    best_nll = np.inf
    
    # Strict success criteria first
    for x0 in starts:
        try:
            opt = minimize(nll_adaptive_stag, x0, args=(y,), bounds=bounds,
                           method="L-BFGS-B", options={"maxiter": 5000})
            if opt.success and opt.fun < len(y) * 40:  # Strict threshold
                if opt.fun < best_nll:
                    best_nll = opt.fun
                    best_result = opt.x.copy()
        except:
            pass
    
    # If no strict solution, try more relaxed criteria
    if best_result is None:
        for x0 in starts:
            try:
                opt = minimize(nll_adaptive_stag, x0, args=(y,), bounds=bounds,
                               method="L-BFGS-B", options={"maxiter": 5000})
                # Relaxed but not too loose
                if opt.fun < len(y) * 60 and not np.any(np.isnan(opt.x)):
                    if opt.fun < best_nll:
                        best_nll = opt.fun
                        best_result = opt.x.copy()
            except:
                pass
    
    return best_result


# =============================================================================
# 1. EVALUATION METRICS
# =============================================================================

def log_likelihood(y, pdf_func, params):
    """Log-likelihood. Higher = better fit. Use sum for cross-sample comparability."""
    try:
        pdf = pdf_func(y, *params)
        pdf = np.clip(pdf, 1e-15, None)
        return float(np.sum(np.log(pdf)))
    except Exception:
        return -np.inf


def pit_uniformity_chi2(y, cdf_func, params, n_bins=10):
    """
    PIT uniformity test (using chi-square instead of variance).
    F(y_i) should follow Uniform(0,1). Returns chi-square stat, lower is better.
    """
    try:
        pit = cdf_func(y, *params)
        pit = np.clip(pit, 1e-10, 1 - 1e-10)
        hist, _ = np.histogram(pit, bins=n_bins, range=(0, 1))
        expected = len(y) / n_bins
        chi2 = np.sum((hist - expected) ** 2 / expected)
        return float(chi2)
    except Exception:
        return np.inf


def wasserstein_1(y, cdf_func, params, n_grid=2000):
    """
    Wasserstein-1 distance.
    Measures how much probability mass needs to move for theoretical distribution to match empirical.
    Lower = better fit.
    """
    try:
        y_sorted = np.sort(y)
        emp_cdf = np.arange(1, len(y) + 1) / len(y)
        x_grid = np.linspace(y.min(), y.max(), n_grid)
        theo_cdf = cdf_func(x_grid, *params)
        emp_cdf_interp = np.interp(x_grid, y_sorted, emp_cdf, left=0.0, right=1.0)
        return float(np.trapz(np.abs(emp_cdf_interp - theo_cdf), x_grid))
    except Exception:
        return np.inf


# =============================================================================
# 2. COMPOSITE SCORING SYSTEM (Percentile-based)
# =============================================================================

def percentile_score(values, higher_is_better=True):
    """Convert scalar values to 0-100 percentile scores."""
    values = np.asarray(values, dtype=float)
    if higher_is_better:
        return np.array([np.mean(values <= v) * 100 for v in values])
    else:
        return np.array([np.mean(values >= v) * 100 for v in values])


def compute_composite_score(metrics_per_model, weights=None):
    """
    Percentile-based composite scoring.
    metrics_per_model: {model_name: {'LL': x, 'W1': x, 'PIT_Chi2': x}}
    """
    if weights is None:
        weights = {'LL': 0.35, 'W1': 0.30, 'PIT_Chi2': 0.35}

    models = list(metrics_per_model.keys())
    ll_vals = np.array([metrics_per_model[m]['LL'] for m in models])
    w1_vals = np.array([metrics_per_model[m]['W1'] for m in models])
    pit_vals = np.array([metrics_per_model[m]['PIT_Chi2'] for m in models])

    pct_ll = percentile_score(ll_vals, higher_is_better=True)
    pct_w1 = percentile_score(w1_vals, higher_is_better=False)
    pct_pit = percentile_score(pit_vals, higher_is_better=False)

    composite = (
        pct_ll * weights['LL'] +
        pct_w1 * weights['W1'] +
        pct_pit * weights['PIT_Chi2']
    )

    df = pd.DataFrame({
        'Model': models,
        'LL': ll_vals, 'W1': w1_vals, 'PIT_Chi2': pit_vals,
        'P_LL': pct_ll, 'P_W1': pct_w1, 'P_PIT': pct_pit,
        'Composite': composite
    })
    df['Rank'] = df['Composite'].rank(ascending=False, method='min').astype(int)
    return df.sort_values('Composite', ascending=False)


# =============================================================================
# 3. TRADITIONAL DISTRIBUTION FITTING (Unified interface)
# =============================================================================

def fit_gaussian(y):
    try:
        m, s = norm.fit(y)
        return {"params": (m, s), "pdf": norm.pdf, "cdf": norm.cdf, "success": True}
    except Exception:
        return {"success": False}


def fit_studentt(y):
    try:
        df, loc, sc = t.fit(y)
        return {"params": (df, loc, sc), "pdf": t.pdf, "cdf": t.cdf, "success": True}
    except Exception:
        return {"success": False}


def fit_beta(y):
    try:
        yc = np.clip(y, 1e-5, 1 - 1e-5)
        a, b_, _, _ = beta.fit(yc, floc=0, fscale=1)
        return {"params": (a, b_, 0, 1), "pdf": beta.pdf, "cdf": beta.cdf, "success": True}
    except Exception:
        return {"success": False}


def fit_gamma(y):
    try:
        a, loc, sc = gamma.fit(y, floc=0)
        return {"params": (a, 0, sc), "pdf": gamma.pdf, "cdf": gamma.cdf, "success": True}
    except Exception:
        return {"success": False}


def fit_weibull(y):
    try:
        shape, loc, sc = weibull_min.fit(y, floc=0)
        return {"params": (shape, 0, sc), "pdf": weibull_min.pdf, "cdf": weibull_min.cdf, "success": True}
    except Exception:
        return {"success": False}


def fit_lognormal(y):
    try:
        s, loc, sc = lognorm.fit(y, floc=0)
        return {"params": (s, 0, sc), "pdf": lognorm.pdf, "cdf": lognorm.cdf, "success": True}
    except Exception:
        return {"success": False}


# Unified fitters dictionary
FITTERS = {
    "Gaussian": fit_gaussian,
    "StudentT": fit_studentt,
    "Gamma": fit_gamma,
    "Beta": fit_beta,
    "Lognormal": fit_lognormal,
}


# =============================================================================
# 4. 10 DATASETS
# =============================================================================
def get_10_datasets():
    datasets = {}
    n = 500

    def f1():
        x = np.random.randn(n) * 0.15 + 0.5
        x[x < 0] *= 0.2
        x[x > 1] = 1 - (x[x > 1] - 1) * 0.2
        return np.clip(x, 0, 1)
    datasets["1_AsymmetricTail_Sensor"] = f1

    def f2():
        x = np.random.gamma(2, 0.2, n)
        x[x > 0.8] = 0.8
        return x
    datasets["2_RightTruncated_Delay"] = f2

    def f3():
        x = np.random.gamma(3, 0.15, n) + 0.2
        x[x < 0.2] = 0.2
        return x
    datasets["3_LeftTruncated_Strength"] = f3

    def f4():
        x = np.concatenate([np.random.normal(0.4, 0.08, n // 2),
                            np.random.normal(0.6, 0.08, n // 2)])
        return np.clip(x, 0, 1)
    datasets["4_Platypus_CPU"] = f4

    def f5():
        x = np.random.beta(3, 3.5, n)
        x[np.random.choice(n, 30, replace=False)] += np.random.uniform(0.1, 0.3, 30)
        return np.clip(x, 0, 1)
    datasets["5_Outliers_Sensor"] = f5

    datasets["6_BoundaryDensity_Saturation"] = lambda: np.clip(np.random.beta(0.8, 4, n), 0, 1)

    def f7():
        x = weibull_min.rvs(1.2, size=n)
        x = np.clip(x, 0, 1.5)
        x[x > 0.6] = x[x > 0.6] * 0.5 + 0.6
        return np.clip(x, 0, 1)
    datasets["7_SingleSideMutation_LongTail"] = f7

    datasets["8_RightSkew_Peak"] = lambda: np.clip(np.random.beta(2, 8, n), 0, 1)

    def f9():
        x = np.random.normal(0.5, 0.2, n)
        x = np.clip(x, 0.2, 0.8)
        return x
    datasets["9_CenterTruncated"] = f9

    def f10():
        x = np.hstack([np.random.normal(0.3, 0.08, 350),
                       np.random.normal(0.7, 0.25, 150)])
        return np.clip(x, 0, 1)
    datasets["10_SingleSideExpand_Tail"] = f10

    return datasets


# =============================================================================
# 5. MAIN EXPERIMENT
# =============================================================================
if __name__ == "__main__":
    np.random.seed(123)
    
    data = get_10_datasets()
    all_models = ["Gaussian", "StudentT", "Gamma",
                  "Beta", "Lognormal", "STAG"]
    n_models = len(all_models)

    # Store average metrics for each dataset
    dataset_metrics = {}
    total_success = 0
    total_tests = len(data) * 30 * n_models  # datasets * repeats * models

    print("=" * 90)
    print("10 datasets | 30 runs each | Metrics: LL, W1, PIT_Chi2")
    print("=" * 90)

    for dname, func in data.items():
        # Collect average metrics for this dataset
        avg_metrics = {}
        raw_metrics = {m: {"LL": [], "PIT_Chi2": [], "W1": [], "cnt": 0} for m in all_models}

        for seed in range(30):
            np.random.seed(seed)
            y = func()
            y = (y - y.min()) / (y.max() - y.min() + 1e-10)
            y = np.clip(y, 1e-5, 1 - 1e-5)

            # --- Traditional distributions ---
            for m_name, fitter in FITTERS.items():
                res = fitter(y)
                if not res["success"]:
                    continue
                params = res["params"]
                pdf_f = res["pdf"]
                cdf_f = res["cdf"]

                ll = log_likelihood(y, pdf_f, params)
                pit = pit_uniformity_chi2(y, cdf_f, params)
                w1 = wasserstein_1(y, cdf_f, params)

                raw_metrics[m_name]["LL"].append(ll)
                raw_metrics[m_name]["PIT_Chi2"].append(pit)
                raw_metrics[m_name]["W1"].append(w1)
                raw_metrics[m_name]["cnt"] += 1
                total_success += 1

            # --- STAG ---
            params = fit_stag(y)
            if params is not None:
                ll = log_likelihood(y, stag_pdf, params)
                pit = pit_uniformity_chi2(y, stag_cdf, params)
                w1 = wasserstein_1(y, stag_cdf, params)

                raw_metrics["STAG"]["LL"].append(ll)
                raw_metrics["STAG"]["PIT_Chi2"].append(pit)
                raw_metrics["STAG"]["W1"].append(w1)
                raw_metrics["STAG"]["cnt"] += 1
                total_success += 1

        # Calculate average metrics for this dataset
        for m in all_models:
            cnt = raw_metrics[m]["cnt"]
            if cnt > 0:
                avg_metrics[m] = {
                    "LL": np.mean(raw_metrics[m]["LL"]),
                    "PIT_Chi2": np.mean(raw_metrics[m]["PIT_Chi2"]),
                    "W1": np.mean(raw_metrics[m]["W1"]),
                }

        if len(avg_metrics) >= 2:
            ranking = compute_composite_score(avg_metrics)
            dataset_metrics[dname] = ranking
            print("\n[%s]" % dname)
            print(ranking[['Model', 'LL', 'W1', 'PIT_Chi2', 'Composite', 'Rank']].to_string(index=False))

    # -------------------------------------------------------------------------
    # Cross-dataset comprehensive ranking
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("Cross-Dataset Comprehensive Ranking (Percentile Method)")
    print("=" * 90)

    composite_per_model = {m: [] for m in all_models}
    for dname, ranking in dataset_metrics.items():
        for _, row in ranking.iterrows():
            composite_per_model[row['Model']].append(row['Composite'])

    final_summary = []
    for m in all_models:
        scores = composite_per_model[m]
        if scores:
            final_summary.append({
                'Model': m,
                'Avg_Composite': np.mean(scores),
                'Std_Composite': np.std(scores),
                'N_Datasets': len(scores),
                'Best_Count': sum(1 for s in scores if s >= 95),
                'Top3_Count': sum(1 for s in scores if s >= 50),
            })

    final_df = pd.DataFrame(final_summary)
    final_df['Final_Rank'] = final_df['Avg_Composite'].rank(
        ascending=False, method='min').astype(int)
    final_df = final_df.sort_values('Avg_Composite', ascending=False)

    print("\n" + final_df[['Model', 'Avg_Composite', 'Std_Composite',
                            'N_Datasets', 'Best_Count', 'Top3_Count', 'Final_Rank']]
          .to_string(index=False))

    # Convergence rate
    conv_rate = total_success / total_tests * 100
    print(f"\nTotal experiments: {total_tests} | Success: {total_success} | Conv rate: {conv_rate:.1f}%")

    print("\n" + "=" * 90)
    print("Metrics explanation:")
    print("  LL       : Log-likelihood (higher is better)")
    print("  W1       : Wasserstein-1 distance (lower is better)")
    print("  PIT_Chi2 : PIT uniformity chi-square (lower is better, gold standard for distribution fitting)")
    print("  Composite: Percentile composite score (0-100, full score = first in all metrics)")
    print("=" * 90)
