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


def simulate_losses(N, v, mean_loss, rng):
    """Cumulative losses and breaches starting from N (shape: (n_steps+1, n_paths)).
    Each attack is a breach with probability v; exponential loss with an average of mean_loss."""
    dN = np.round(np.diff(N, axis=0)).astype(int)
    breaches = rng.binomial(dN, v)
    shape = np.where(breaches > 0, breaches, 1)
    dL = np.where(breaches > 0, rng.gamma(shape, mean_loss), 0.0)
    pad = np.zeros((1, N.shape[1]))
    return np.vstack([pad, np.cumsum(breaches, axis=0)]), np.vstack([pad, np.cumsum(dL, axis=0)])
