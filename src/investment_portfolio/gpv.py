"""Gaussian Polynomial Volatility (GPV) model for option pricing.

Implements the stochastic volatility model from Bonesini, Callegaro & Grasselli
(arXiv:2212.08297v2) where spot volatility is a polynomial function of a Gaussian
Volterra process.  Supports exponential (Markovian), fractional, log-modulated, and
shifted-fractional kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import assert_never

import numpy as np
from scipy.integrate import quad

from investment_portfolio._validation import validate_finite

__all__ = [
    "KERNEL_TYPES",
    "ForwardVarianceCurve",
    "GPVModelConfig",
    "GPVOptionPrice",
    "GPVSimulationResult",
    "KernelType",
    "VIXResult",
    "compute_vix_squared",
    "forward_variance",
    "kernel_function",
    "kernel_variance",
    "polynomial_volatility",
    "price_european_gpv",
    "simulate_gaussian_process",
    "simulate_paths",
]


_NORMALIZATION_FLOOR: float = 1e-15
_ZERO_DECAY_TOL: float = 1e-12


# ── Enum & Registry ────────────────────────────────────────────────────────


class KernelType(StrEnum):
    """Kernel type for the Gaussian Volterra process."""

    EXPONENTIAL = auto()  # Markovian, best performer
    FRACTIONAL = auto()  # K(t) = t^{H-1/2}
    LOG_MODULATED = auto()  # K(t) = t^{H-1/2} * max(theta*log(1/t), 1)^{-beta}
    SHIFTED_FRACTIONAL = auto()  # K(t) = (t+epsilon)^{H-1/2}


KERNEL_TYPES: dict[str, str] = {
    "Exponential (Markovian)": "exponential",
    "Fractional": "fractional",
    "Log-modulated": "log_modulated",
    "Shifted Fractional": "shifted_fractional",
}


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GPVModelConfig:
    """Configuration for the GPV model."""

    kernel_type: KernelType
    hurst: float
    poly_coeffs: tuple[float, ...]
    xi_0: np.ndarray
    maturities: np.ndarray
    rho: float = -0.7
    epsilon: float = 0.1
    theta: float = 1.0
    beta: float = 0.5


@dataclass(frozen=True, slots=True)
class ForwardVarianceCurve:
    """Forward variance curve conditioned on information at time t."""

    maturities: np.ndarray
    values: np.ndarray
    conditioning_time: float


@dataclass(frozen=True, slots=True)
class GPVSimulationResult:
    """Result of a GPV path simulation."""

    time_grid: np.ndarray
    price_paths: np.ndarray
    volatility_paths: np.ndarray
    gaussian_paths: np.ndarray


@dataclass(frozen=True, slots=True)
class VIXResult:
    """VIX computation result."""

    vix_values: np.ndarray
    vix_squared: np.ndarray
    observation_time: float


@dataclass(frozen=True, slots=True)
class GPVOptionPrice:
    """European option price from GPV Monte Carlo."""

    call: float
    put: float
    call_std_error: float
    put_std_error: float
    num_simulations: int


# ── Internal Helpers ───────────────────────────────────────────────────────


def _double_factorial(*, n: int) -> int:
    """Compute the double factorial n!! = n * (n-2) * ... * 1.

    Args:
        n: Non-negative integer.

    Returns:
        The double factorial of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        msg = "n must be non-negative for double factorial"
        raise ValueError(msg)
    if n <= 1:
        return 1
    return math.prod(range(n, 0, -2))


def _gaussian_moments(*, variance: float, order: int) -> float:
    """Compute E[G^order] for a zero-mean Gaussian with given variance.

    Args:
        variance: Variance of the Gaussian random variable.
        order: Non-negative integer moment order.

    Returns:
        The raw moment E[G^order].

    Raises:
        ValueError: If order is negative or variance is negative.
    """
    if order < 0:
        msg = "order must be non-negative"
        raise ValueError(msg)
    if variance < 0:
        msg = "variance must be non-negative"
        raise ValueError(msg)
    if order == 0:
        return 1.0
    if order % 2 == 1:
        return 0.0
    return variance ** (order // 2) * _double_factorial(n=order - 1)


def _discrete_convolution(*, coeffs: tuple[float, ...]) -> np.ndarray:
    """Compute the discrete self-convolution of polynomial coefficients.

    Args:
        coeffs: Polynomial coefficients (alpha_0, alpha_1, ..., alpha_M).

    Returns:
        Convolution coefficients of length 2*M + 1.
    """
    arr = np.array(coeffs, dtype=np.float64)
    return np.convolve(arr, arr)


def _normalization_factor(
    *, poly_coeffs: tuple[float, ...], kernel_variance_value: float
) -> float:
    """Compute g(u) = E[p(X_u)^2] for normalization.

    Uses the convolution of polynomial coefficients and Gaussian moments.

    Args:
        poly_coeffs: Polynomial coefficients.
        kernel_variance_value: The kernel variance integral value.

    Returns:
        The normalization factor g(u).
    """
    conv = _discrete_convolution(coeffs=poly_coeffs)
    result = 0.0
    for k, c_k in enumerate(conv):
        result += c_k * _gaussian_moments(variance=kernel_variance_value, order=k)
    return result


def _validate_gpv_config(*, config: GPVModelConfig) -> None:
    """Validate GPV model configuration.

    Args:
        config: The model configuration to validate.

    Raises:
        ValueError: If any field is out of valid range.
    """
    if not (0.0 < config.hurst < 1.0):
        msg = "hurst must be in (0, 1)"
        raise ValueError(msg)
    if len(config.poly_coeffs) == 0:
        msg = "poly_coeffs must be non-empty"
        raise ValueError(msg)
    if config.xi_0.ndim != 1 or len(config.xi_0) == 0:
        msg = "xi_0 must be a non-empty 1-D array"
        raise ValueError(msg)
    validate_finite(arr=config.xi_0, name="xi_0")
    validate_finite(arr=config.maturities, name="maturities")
    if config.maturities.ndim != 1 or len(config.maturities) == 0:
        msg = "maturities must be a non-empty 1-D array"
        raise ValueError(msg)
    if len(config.xi_0) != len(config.maturities):
        msg = "xi_0 and maturities must have the same length"
        raise ValueError(msg)
    if not (-1.0 <= config.rho <= 1.0):
        msg = "rho must be in [-1, 1]"
        raise ValueError(msg)
    if config.epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)


# ── Public API ─────────────────────────────────────────────────────────────


def kernel_function(
    *,
    t: np.ndarray,
    kernel_type: KernelType,
    hurst: float = 0.5,
    epsilon: float = 0.1,
    theta: float = 1.0,
    beta: float = 0.5,
) -> np.ndarray:
    """Evaluate the Volterra kernel K(t) for the given kernel type.

    Args:
        t: Time points (non-negative).
        kernel_type: Which kernel to use.
        hurst: Hurst parameter H in (0, 1).
        epsilon: Scale parameter for exponential / shifted-fractional kernels.
        theta: Modulation parameter for the log-modulated kernel.
        beta: Decay exponent for the log-modulated kernel.

    Returns:
        Array of kernel values K(t).

    Raises:
        ValueError: If hurst is out of range or epsilon is non-positive.
    """
    if not (0.0 < hurst < 1.0):
        msg = "hurst must be in (0, 1)"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)

    t = np.asarray(t, dtype=np.float64)
    h = hurst - 0.5

    match kernel_type:
        case KernelType.EXPONENTIAL:
            c = (0.5 - hurst) / epsilon
            return epsilon**h * np.exp(-c * t)

        case KernelType.FRACTIONAL:
            result = np.zeros_like(t)
            mask = t > 0
            result[mask] = t[mask] ** h
            return result

        case KernelType.LOG_MODULATED:
            result = np.zeros_like(t)
            mask = t > 0
            log_factor = np.maximum(theta * np.log(1.0 / t[mask]), 1.0)
            result[mask] = t[mask] ** h * log_factor ** (-beta)
            return result

        case KernelType.SHIFTED_FRACTIONAL:
            return (t + epsilon) ** h

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def kernel_variance(
    *,
    upper_limit: float,
    kernel_type: KernelType,
    hurst: float = 0.5,
    epsilon: float = 0.1,
    theta: float = 1.0,
    beta: float = 0.5,
) -> float:
    """Compute the kernel variance integral int_0^T K(s)^2 ds.

    Uses closed-form expressions when available (exponential, fractional),
    otherwise falls back to numerical quadrature.

    Args:
        upper_limit: Upper integration limit T (non-negative).
        kernel_type: Which kernel to use.
        hurst: Hurst parameter H in (0, 1).
        epsilon: Scale parameter for exponential / shifted-fractional kernels.
        theta: Modulation parameter for the log-modulated kernel.
        beta: Decay exponent for the log-modulated kernel.

    Returns:
        The integral value.

    Raises:
        ValueError: If upper_limit is negative, hurst out of range, or
            epsilon non-positive.
    """
    if upper_limit < 0:
        msg = "upper_limit must be non-negative"
        raise ValueError(msg)
    if not (0.0 < hurst < 1.0):
        msg = "hurst must be in (0, 1)"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)

    if upper_limit == 0.0:
        return 0.0

    h = hurst - 0.5

    match kernel_type:
        case KernelType.EXPONENTIAL:
            c = (0.5 - hurst) / epsilon
            if abs(c) < _ZERO_DECAY_TOL:
                # H=0.5 limit: K(t)=1, variance = T
                return float(upper_limit)
            two_c = 2.0 * c
            return float(
                epsilon ** (2.0 * h) / two_c * (1.0 - math.exp(-two_c * upper_limit))
            )

        case KernelType.FRACTIONAL:
            two_h = 2.0 * hurst
            return float(upper_limit**two_h / two_h)

        case KernelType.LOG_MODULATED | KernelType.SHIFTED_FRACTIONAL:

            def _integrand(s: float) -> float:
                k = kernel_function(
                    t=np.array([s]),
                    kernel_type=kernel_type,
                    hurst=hurst,
                    epsilon=epsilon,
                    theta=theta,
                    beta=beta,
                )
                return float(k[0] ** 2)

            result, _ = quad(_integrand, 0, upper_limit)
            return float(result)

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def polynomial_volatility(
    *,
    x: np.ndarray,
    poly_coeffs: tuple[float, ...],
) -> np.ndarray:
    """Evaluate the polynomial volatility function p(x).

    Uses Horner's scheme.

    Args:
        x: Input values (Gaussian process values).
        poly_coeffs: Polynomial coefficients (alpha_0, alpha_1, ..., alpha_M).

    Returns:
        Array of p(x) values.

    Raises:
        ValueError: If poly_coeffs is empty.
    """
    if len(poly_coeffs) == 0:
        msg = "poly_coeffs must be non-empty"
        raise ValueError(msg)
    x_arr = np.asarray(x, dtype=np.float64)
    # Horner's scheme: p(x) = a0 + x*(a1 + x*(a2 + ...))
    result = np.full_like(x_arr, poly_coeffs[-1], dtype=np.float64)
    for c in reversed(poly_coeffs[:-1]):
        result = result * x_arr + c
    return result


def forward_variance(
    *,
    config: GPVModelConfig,
    conditioning_time: float,
    gaussian_values: np.ndarray | float | None = None,
) -> ForwardVarianceCurve:
    """Compute the forward variance curve xi_t(u) conditioned at time t.

    At time t=0 with no conditioning, xi_0(u) is returned directly from the
    initial curve.  For t>0, the formula uses convolution of polynomial
    coefficients, binomial expansion, and Gaussian moments.

    When *gaussian_values* is an array of per-simulation values, their mean
    is used as a single conditioning level (mean-field approximation).  This
    is appropriate for aggregate diagnostics such as VIX computation.  Pass
    a scalar ``float`` to condition on a specific Gaussian process value.

    Args:
        config: GPV model configuration.
        conditioning_time: The conditioning time t.
        gaussian_values: Conditioning value for the Gaussian process at time
            *t*.  A scalar conditions on that exact value; an array is reduced
            to its mean (mean-field approximation).  ``None`` returns the
            unconditional initial curve.

    Returns:
        A ForwardVarianceCurve with maturities and forward variance values.

    Raises:
        ValueError: If config is invalid.
    """
    _validate_gpv_config(config=config)

    if conditioning_time == 0.0 or gaussian_values is None:
        return ForwardVarianceCurve(
            maturities=config.maturities.copy(),
            values=config.xi_0.copy(),
            conditioning_time=conditioning_time,
        )

    conv = _discrete_convolution(coeffs=config.poly_coeffs)
    n_maturities = len(config.maturities)

    # Scalar → use directly; array → mean-field approximation
    z = (
        float(gaussian_values)
        if np.ndim(gaussian_values) == 0
        else float(np.mean(gaussian_values))
    )
    values = np.empty(n_maturities, dtype=np.float64)

    for idx in range(n_maturities):
        u = config.maturities[idx]
        if u <= conditioning_time:
            values[idx] = config.xi_0[idx]
            continue

        kv = kernel_variance(
            upper_limit=u,
            kernel_type=config.kernel_type,
            hurst=config.hurst,
            epsilon=config.epsilon,
            theta=config.theta,
            beta=config.beta,
        )
        g_u = _normalization_factor(
            poly_coeffs=config.poly_coeffs,
            kernel_variance_value=kv,
        )

        if g_u < _NORMALIZATION_FLOOR:
            values[idx] = config.xi_0[idx]
            continue

        # Residual variance for conditional distribution
        kv_t = kernel_variance(
            upper_limit=conditioning_time,
            kernel_type=config.kernel_type,
            hurst=config.hurst,
            epsilon=config.epsilon,
            theta=config.theta,
            beta=config.beta,
        )
        residual_var = max(kv - kv_t, 0.0)

        # Compute sum using convolution + binomial expansion
        total = 0.0
        for k, c_k in enumerate(conv):
            binom_sum = 0.0
            for i in range(k + 1):
                if (k - i) % 2 == 1 and z == 0.0:
                    continue
                binom_coeff = math.comb(k, i)
                moment = _gaussian_moments(variance=residual_var, order=i)
                binom_sum += binom_coeff * (z ** (k - i)) * moment
            total += c_k * binom_sum

        values[idx] = config.xi_0[idx] / g_u * total

    # Ensure non-negative
    values = np.maximum(values, 0.0)

    return ForwardVarianceCurve(
        maturities=config.maturities.copy(),
        values=values,
        conditioning_time=conditioning_time,
    )


def compute_vix_squared(
    *,
    config: GPVModelConfig,
    observation_time: float,
    gaussian_values: np.ndarray | float | None = None,
    num_quadrature_points: int = 100,
) -> VIXResult:
    """Compute VIX^2 at the observation time.

    VIX^2_T = (100^2 / Delta) * int_T^{T+Delta} xi_T(u) du
    where Delta = 30/365.

    Args:
        config: GPV model configuration.
        observation_time: The observation time T.
        gaussian_values: Gaussian process values at observation time.
        num_quadrature_points: Number of points for trapezoidal integration.

    Returns:
        A VIXResult with VIX values and VIX squared.

    Raises:
        ValueError: If config is invalid.
    """
    _validate_gpv_config(config=config)

    delta = 30.0 / 365.0

    fwd = forward_variance(
        config=config,
        conditioning_time=observation_time,
        gaussian_values=gaussian_values,
    )

    # Build integration grid within [T, T+Delta]
    t_start = observation_time
    t_end = observation_time + delta
    quad_grid = np.linspace(t_start, t_end, num_quadrature_points)

    # Interpolate forward variance onto integration grid
    fwd_interp = np.interp(quad_grid, fwd.maturities, fwd.values)

    # Trapezoidal integration
    integral = float(np.trapezoid(fwd_interp, quad_grid))

    vix_sq = 100.0**2 / delta * integral
    vix_sq = max(vix_sq, 0.0)
    vix_val = math.sqrt(vix_sq)

    return VIXResult(
        vix_values=np.array([vix_val]),
        vix_squared=np.array([vix_sq]),
        observation_time=observation_time,
    )


def simulate_gaussian_process(
    *,
    time_grid: np.ndarray,
    kernel_type: KernelType,
    hurst: float = 0.5,
    epsilon: float = 0.1,
    theta: float = 1.0,
    beta: float = 0.5,
    num_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate the Gaussian Volterra process X_t on a given time grid.

    For the exponential (Markovian) kernel, uses an efficient AR(1)
    recursion.  For other kernels, uses Cholesky decomposition of the
    covariance matrix.

    Args:
        time_grid: Sorted array of time points starting at 0.
        kernel_type: Which kernel to use.
        hurst: Hurst parameter H in (0, 1).
        epsilon: Scale parameter.
        theta: Modulation parameter (log-modulated kernel).
        beta: Decay exponent (log-modulated kernel).
        num_simulations: Number of independent paths to simulate.
        rng: NumPy random Generator for reproducibility.

    Returns:
        Array of shape (len(time_grid), num_simulations) with process values.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not (0.0 < hurst < 1.0):
        msg = "hurst must be in (0, 1)"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    if num_simulations <= 0:
        msg = "num_simulations must be positive"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    time_grid = np.asarray(time_grid, dtype=np.float64)
    n_steps = len(time_grid)
    paths = np.zeros((n_steps, num_simulations), dtype=np.float64)

    if n_steps <= 1:
        return paths

    if kernel_type == KernelType.EXPONENTIAL:
        # AR(1) recursion: X_{t+dt} = exp(-c*dt) * X_t + sigma * Z
        c = (0.5 - hurst) / epsilon
        for i in range(1, n_steps):
            dt = time_grid[i] - time_grid[i - 1]
            if abs(c) < _ZERO_DECAY_TOL:
                # H=0.5 limit: standard BM, X_{t+dt}=X_t+sqrt(dt)*Z
                sigma_inc = math.sqrt(dt)
                paths[i] = paths[i - 1] + sigma_inc * rng.standard_normal(
                    num_simulations
                )
            else:
                decay = math.exp(-c * dt)
                h = hurst - 0.5
                var_inc = (
                    epsilon ** (2.0 * h) / (2.0 * c) * (1.0 - math.exp(-2.0 * c * dt))
                )
                sigma_inc = math.sqrt(max(var_inc, 0.0))
                z = rng.standard_normal(num_simulations)
                paths[i] = decay * paths[i - 1] + sigma_inc * z
    else:
        # Cholesky decomposition approach
        # Build covariance matrix: Cov(X_s, X_t) = int_0^{min(s,t)} K(s-u)K(t-u) du
        cov_matrix = np.zeros((n_steps, n_steps), dtype=np.float64)
        for i in range(n_steps):
            for j in range(i, n_steps):
                s, t_j = time_grid[i], time_grid[j]
                if s == 0.0 or t_j == 0.0:
                    cov_matrix[i, j] = 0.0
                    cov_matrix[j, i] = 0.0
                    continue

                min_st = min(s, t_j)
                n_quad = 50

                quad_pts = np.linspace(0, min_st, n_quad + 1)[1:]
                k_s = kernel_function(
                    t=s - quad_pts,
                    kernel_type=kernel_type,
                    hurst=hurst,
                    epsilon=epsilon,
                    theta=theta,
                    beta=beta,
                )
                k_t = kernel_function(
                    t=t_j - quad_pts,
                    kernel_type=kernel_type,
                    hurst=hurst,
                    epsilon=epsilon,
                    theta=theta,
                    beta=beta,
                )
                du = min_st / n_quad
                val = float(np.sum(k_s * k_t) * du)
                cov_matrix[i, j] = val
                cov_matrix[j, i] = val

        # Add ridge scaled to matrix magnitude for numerical stability
        trace_scale = max(1e-8, 1e-6 * np.trace(cov_matrix) / n_steps)
        cov_matrix += trace_scale * np.eye(n_steps)

        try:
            chol = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback: eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals = np.maximum(eigvals, trace_scale)
            cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
            try:
                chol = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                # Last resort: SVD-based sampling
                chol = eigvecs @ np.diag(np.sqrt(eigvals))

        z = rng.standard_normal((n_steps, num_simulations))
        paths = chol @ z

    return paths


def simulate_paths(
    *,
    config: GPVModelConfig,
    spot: float,
    time_horizon: float,
    num_steps: int = 100,
    num_simulations: int = 10_000,
    risk_free_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> GPVSimulationResult:
    """Simulate asset price and volatility paths under the GPV model.

    Uses Euler-Maruyama discretization in log-space with correlated
    Brownian motions.  When *risk_free_rate* is non-zero the drift
    includes the risk-neutral term ``r * dt``, making the paths suitable
    for derivative pricing via Monte Carlo.

    Args:
        config: GPV model configuration.
        spot: Initial asset price (must be positive).
        time_horizon: Simulation horizon in years (must be positive).
        num_steps: Number of time steps.
        num_simulations: Number of simulation paths.
        risk_free_rate: Annualized risk-free rate for the risk-neutral
            drift (default 0 gives physical-measure dynamics).
        rng: NumPy random Generator for reproducibility.

    Returns:
        A GPVSimulationResult with all simulated paths.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_gpv_config(config=config)

    if spot <= 0:
        msg = "spot must be positive"
        raise ValueError(msg)
    if time_horizon <= 0:
        msg = "time_horizon must be positive"
        raise ValueError(msg)
    if num_steps <= 0:
        msg = "num_steps must be positive"
        raise ValueError(msg)
    if num_simulations <= 0:
        msg = "num_simulations must be positive"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    time_grid = np.linspace(0, time_horizon, num_steps + 1)
    dt = time_horizon / num_steps

    # Simulate Gaussian process
    gaussian_paths = simulate_gaussian_process(
        time_grid=time_grid,
        kernel_type=config.kernel_type,
        hurst=config.hurst,
        epsilon=config.epsilon,
        theta=config.theta,
        beta=config.beta,
        num_simulations=num_simulations,
        rng=rng,
    )

    # Pre-compute normalization factors and interpolated xi_0
    xi_0_interp = np.interp(time_grid, config.maturities, config.xi_0)

    # Compute volatility at each time step
    vol_paths = np.zeros((num_steps + 1, num_simulations), dtype=np.float64)
    for i in range(num_steps + 1):
        t_i = time_grid[i]
        kv = kernel_variance(
            upper_limit=max(t_i, 1e-10),
            kernel_type=config.kernel_type,
            hurst=config.hurst,
            epsilon=config.epsilon,
            theta=config.theta,
            beta=config.beta,
        )
        g_val = _normalization_factor(
            poly_coeffs=config.poly_coeffs,
            kernel_variance_value=kv,
        )

        p_vals = polynomial_volatility(
            x=gaussian_paths[i],
            poly_coeffs=config.poly_coeffs,
        )

        base_vol = np.sqrt(np.maximum(xi_0_interp[i], 0.0))
        if g_val > _NORMALIZATION_FLOOR:
            vol_paths[i] = base_vol * p_vals / math.sqrt(g_val)
        else:
            vol_paths[i] = base_vol
        # Floor at zero: polynomial p(x) can be negative in the tails
        vol_paths[i] = np.maximum(vol_paths[i], 0.0)

    # Euler-Maruyama in log-space
    log_prices = np.full(
        (num_steps + 1, num_simulations), math.log(spot), dtype=np.float64
    )
    rho = config.rho
    rho_bar = math.sqrt(1.0 - rho**2)
    sqrt_dt = math.sqrt(dt)

    for i in range(num_steps):
        sigma = vol_paths[i]
        dw = rng.standard_normal(num_simulations) * sqrt_dt
        db_perp = rng.standard_normal(num_simulations) * sqrt_dt
        db = rho * dw + rho_bar * db_perp

        log_prices[i + 1] = (
            log_prices[i] + (risk_free_rate - 0.5 * sigma**2) * dt + sigma * db
        )

    price_paths = np.exp(log_prices)

    return GPVSimulationResult(
        time_grid=time_grid,
        price_paths=price_paths,
        volatility_paths=vol_paths,
        gaussian_paths=gaussian_paths,
    )


def price_european_gpv(
    *,
    config: GPVModelConfig,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.0,
    num_steps: int = 100,
    num_simulations: int = 50_000,
    rng: np.random.Generator | None = None,
) -> GPVOptionPrice:
    """Price a European option under the GPV stochastic volatility model.

    Uses Monte Carlo simulation to estimate call and put prices with
    standard errors.

    Args:
        config: GPV model configuration.
        spot: Current asset price (must be positive).
        strike: Option strike price (must be positive).
        time_to_expiry: Time to expiration in years (must be positive).
        risk_free_rate: Annualized risk-free interest rate (non-negative).
        num_steps: Number of Euler-Maruyama time steps.
        num_simulations: Number of Monte Carlo paths.
        rng: NumPy random Generator for reproducibility.

    Returns:
        A GPVOptionPrice with call, put, and standard errors.

    Raises:
        ValueError: If inputs are invalid.
    """
    if spot <= 0:
        msg = "spot must be positive"
        raise ValueError(msg)
    if strike <= 0:
        msg = "strike must be positive"
        raise ValueError(msg)
    if time_to_expiry <= 0:
        msg = "time_to_expiry must be positive"
        raise ValueError(msg)
    if risk_free_rate < 0:
        msg = "risk_free_rate must be non-negative"
        raise ValueError(msg)

    result = simulate_paths(
        config=config,
        spot=spot,
        time_horizon=time_to_expiry,
        num_steps=num_steps,
        num_simulations=num_simulations,
        risk_free_rate=risk_free_rate,
        rng=rng,
    )

    terminal_prices = result.price_paths[-1]
    discount = math.exp(-risk_free_rate * time_to_expiry)

    call_payoffs = np.maximum(terminal_prices - strike, 0.0)
    put_payoffs = np.maximum(strike - terminal_prices, 0.0)

    call_price = float(discount * np.mean(call_payoffs))
    put_price = float(discount * np.mean(put_payoffs))

    n = num_simulations
    call_se = float(discount * np.std(call_payoffs, ddof=1) / math.sqrt(n))
    put_se = float(discount * np.std(put_payoffs, ddof=1) / math.sqrt(n))

    return GPVOptionPrice(
        call=call_price,
        put=put_price,
        call_std_error=call_se,
        put_std_error=put_se,
        num_simulations=num_simulations,
    )
