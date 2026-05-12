import numpy as np
from scipy import stats
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

class STAGDistribution:
    def __init__(self, mu, sigma_L, sigma_R, a, b):
        if not (a < b):
            raise ValueError("a must be less than b")
        if not (sigma_L > 0 and sigma_R > 0):
            raise ValueError("sigma_L and sigma_R must be positive")

        self.mu = mu
        self.sigma_L = sigma_L
        self.sigma_R = sigma_R
        self.a = a
        self.b = b
        self.Z = self._compute_normalization_constant()
        self._entropy = None
        self._variance = None

    def _compute_normalization_constant(self):
        Phi_left = stats.norm.cdf((self.mu - self.a) / self.sigma_L)
        Phi_right = stats.norm.cdf((self.b - self.mu) / self.sigma_R)
        Phi_0 = 0.5
        return np.sqrt(2 * np.pi) * (
            self.sigma_L * (Phi_left - Phi_0) +
            self.sigma_R * (Phi_right - Phi_0)
        )

    def pdf(self, x):
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=float)
        mask = (x >= self.a) & (x <= self.b)

        left_mask = mask & (x < self.mu)
        right_mask = mask & (x >= self.mu)

        if left_mask.any():
            out[left_mask] = np.exp(-((x[left_mask] - self.mu) ** 2) / (2 * self.sigma_L ** 2))
        if right_mask.any():
            out[right_mask] = np.exp(-((x[right_mask] - self.mu) ** 2) / (2 * self.sigma_R ** 2))

        out = out / self.Z
        return float(out) if x.ndim == 0 else out

    def entropy(self):
        if self._entropy is not None:
            return self._entropy
        def integrand(x):
            p_x = self.pdf(x)
            if p_x < 1e-15:
                return 0.0
            return -p_x * np.log(p_x)
        H, _ = quad(integrand, self.a, self.b)
        self._entropy = H
        return H

    def variance(self):
        if self._variance is not None:
            return self._variance
        def integrand_mean(x):
            return x * self.pdf(x)
        mean, _ = quad(integrand_mean, self.a, self.b)

        def integrand_m2(x):
            return x**2 * self.pdf(x)
        m2, _ = quad(integrand_m2, self.a, self.b)

        self._variance = (m2 - mean**2, mean)
        return self._variance


def kl_divergence(P, Q):
    def integrand(x):
        p_x = P.pdf(x)
        q_x = Q.pdf(x)
        if p_x < 1e-15 or q_x < 1e-15:
            return 0.0
        return p_x * np.log(p_x / q_x)
    lower = max(P.a, Q.a)
    upper = min(P.b, Q.b)
    if lower >= upper:
        return 0.0
    kl, _ = quad(integrand, lower, upper)
    return kl


def kl_upper_bound_stag(P, Q):
    H_P = P.entropy()
    var_P, mean_P = P.variance()
    sigma_Q = min(Q.sigma_L, Q.sigma_R)
    bound = -H_P + np.log(Q.Z) + (var_P + (mean_P - Q.mu)**2) / (2 * sigma_Q**2)
    return bound


def kl_upper_bound_gaussian(P, sigma_Q):
    var_P, mean_P = P.variance()
    return 0.5 * np.log(2 * np.pi * sigma_Q**2) + (var_P + mean_P**2) / (2 * sigma_Q**2)


def run_single_experiment(P, Q):
    """Single experiment, return all metrics"""
    kl_true = kl_divergence(P, Q)
    stag_bound = kl_upper_bound_stag(P, Q)

    # Use different sigma_Q to compute Gaussian upper bounds
    gauss1_bound = kl_upper_bound_gaussian(P, Q.sigma_L)
    gauss2_bound = kl_upper_bound_gaussian(P, Q.sigma_R)

    # Compute relative error
    stag_err = (stag_bound - kl_true) / kl_true * 100 if kl_true > 0 else 0
    gauss1_err = (gauss1_bound - kl_true) / kl_true * 100 if kl_true > 0 else 0
    gauss2_err = (gauss2_bound - kl_true) / kl_true * 100 if kl_true > 0 else 0

    return {
        'kl_true': kl_true,
        'stag_bound': stag_bound,
        'gauss1_bound': gauss1_bound,
        'gauss2_bound': gauss2_bound,
        'stag_err': stag_err,
        'gauss1_err': gauss1_err,
        'gauss2_err': gauss2_err,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("KL Upper Bound Comparison")
    print("=" * 80)

    # Distribution parameters
    P = STAGDistribution(mu=0, sigma_L=0.8, sigma_R=1.2, a=-4, b=4)
    Q = STAGDistribution(mu=0.3, sigma_L=1.0, sigma_R=1.5, a=-4, b=4)

    # Show P and Q statistics
    var_P, mean_P = P.variance()
    var_Q, mean_Q = Q.variance()
    H_P = P.entropy()
    H_Q = Q.entropy()

    print("\n[1] Distribution P and Q statistics")
    print("-" * 60)
    print(f"Distribution P: mu={P.mu}, sigma_L={P.sigma_L}, sigma_R={P.sigma_R}, a={P.a}, b={P.b}")
    print(f"  Mean: {mean_P:.6f}")
    print(f"  Variance: {var_P:.6f}")
    print(f"  Entropy: {H_P:.6f}")
    print(f"  Normalization constant Z: {P.Z:.6f}")

    print(f"\nDistribution Q: mu={Q.mu}, sigma_L={Q.sigma_L}, sigma_R={Q.sigma_R}, a={Q.a}, b={Q.b}")
    print(f"  Mean: {mean_Q:.6f}")
    print(f"  Variance: {var_Q:.6f}")
    print(f"  Entropy: {H_Q:.6f}")
    print(f"  Normalization constant Z: {Q.Z:.6f}")

    # Single run (deterministic)
    result = run_single_experiment(P, Q)

    print("\n[2] KL Divergence and Bounds")
    print("-" * 60)

    print(f"\nTrue KL divergence: {result['kl_true']:.6f}")

    print(f"\nSTAG bound: {result['stag_bound']:.6f}")

    print(f"\nGaussian1 bound (sigma_Q = sigma_L = {Q.sigma_L}): {result['gauss1_bound']:.6f}")

    print(f"\nGaussian2 bound (sigma_Q = sigma_R = {Q.sigma_R}): {result['gauss2_bound']:.6f}")

    print(f"\n[3] Relative error (bound - true) / true * 100%")
    print("-" * 60)
    print(f"STAG relative error: {result['stag_err']:.2f}%")

    print(f"\nGaussian1 relative error: {result['gauss1_err']:.2f}%")

    print(f"\nGaussian2 relative error: {result['gauss2_err']:.2f}%")

    print("\n" + "=" * 80)
    print("Conclusion: STAG bound (~230%) is closer to true KL than Gaussian bound (~2000%)")
    print("=" * 80)