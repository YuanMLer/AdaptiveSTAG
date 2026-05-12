# STAG Distribution Experiments

## Files

| File | Description |
|------|-------------|
| `stag_simulation_final.py` | Distribution fitting on 10 datasets |
| `stag_regression_runs.py` | Regression on SECOM dataset |
| `stag_kl_bound_comparison.py` | KL divergence upper bound comparison |
| `requirements.txt` | Python dependencies |

## Requirements

```
pip install -r requirements.txt
```

## Run

```bash
python stag_simulation_final.py
python stag_regression_runs.py
python stag_kl_bound_comparison.py
```

## Experiments

1. **Distribution Fitting**: Compare STAG with Gaussian, StudentT, Gamma, Beta, Lognormal on 10 synthetic datasets. Metrics: LL, W1, PIT_Chi2.

2. **Regression**: STAG regression on SECOM dataset (30 runs).

3. **KL Bound**: Compare STAG upper bound vs Gaussian.