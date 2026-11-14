import numpy as np
from ..iVi.iVi_hawkes import IVIHawkesProcess
from ..kernel.exponential_kernel import ExponentialKernel


def make_cyber_hawkes(alpha, lam0, xi, beta, seed=42):
    """Hawkes stochastic intensity: lambda_t = alpha + (lam0-alpha)e^{-xi t} + beta * sum e^{-xi (t - tau_i)}."""
    kernel = ExponentialKernel(c=beta, lam=xi)
    g0 = lambda t: alpha + (lam0 - alpha) * np.exp(-xi * np.asarray(t))
    g0_bar = lambda t: alpha * np.asarray(t) + (lam0 - alpha) * (1 - np.exp(-xi * np.asarray(t))) / xi
    return IVIHawkesProcess(kernel=kernel, g0_bar=g0_bar, g0=g0, rng=np.random.default_rng(seed))


def mean_N(t, alpha, lam0, xi, beta):
    """E[N_t] from Proposition 1."""
    t = np.asarray(t)
    m = alpha * xi / (xi - beta)
    return m * t - (lam0 - m) / (xi - beta) * (np.exp(-(xi - beta) * t) - 1)
