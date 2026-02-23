"""Tests for the Gaussian Polynomial Volatility (GPV) module."""

import math

import numpy as np
import pytest

from investment_portfolio.gpv import (
    ForwardVarianceCurve,
    GPVModelConfig,
    GPVOptionPrice,
    GPVSimulationResult,
    KernelType,
    VIXResult,
    _discrete_convolution,
    _double_factorial,
    _gaussian_moments,
    _normalization_factor,
    compute_vix_squared,
    forward_variance,
    kernel_function,
    kernel_variance,
    polynomial_volatility,
    price_european_gpv,
    simulate_gaussian_process,
    simulate_paths,
)

# ── Reference GPV config ──────────────────────────────────────────────────

_MATURITIES = np.linspace(0.01, 2.0, 50)
_FLAT_XI = np.full(50, 0.04)  # flat 20% vol => xi_0 = 0.04


def _make_config(
    *,
    kernel_type: KernelType = KernelType.EXPONENTIAL,
    hurst: float = 0.5,
    poly_coeffs: tuple[float, ...] = (1.0,),
    rho: float = -0.7,
    epsilon: float = 0.1,
) -> GPVModelConfig:
    return GPVModelConfig(
        kernel_type=kernel_type,
        hurst=hurst,
        poly_coeffs=poly_coeffs,
        xi_0=_FLAT_XI.copy(),
        maturities=_MATURITIES.copy(),
        rho=rho,
        epsilon=epsilon,
    )


# ── TestDoubleFactorial ──────────────────────────────────────────────────


class TestDoubleFactorial:
    """Tests for the _double_factorial helper."""

    def test_zero(self) -> None:
        """0!! = 1."""
        assert _double_factorial(n=0) == 1

    def test_one(self) -> None:
        """1!! = 1."""
        assert _double_factorial(n=1) == 1

    def test_five(self) -> None:
        """5!! = 5 * 3 * 1 = 15."""
        assert _double_factorial(n=5) == 15

    def test_six(self) -> None:
        """6!! = 6 * 4 * 2 = 48."""
        assert _double_factorial(n=6) == 48

    def test_negative_raises(self) -> None:
        """Negative n raises ValueError."""
        with pytest.raises(ValueError, match="n must be non-negative"):
            _double_factorial(n=-1)


# ── TestGaussianMoments ──────────────────────────────────────────────────


class TestGaussianMoments:
    """Tests for the _gaussian_moments helper."""

    def test_odd_order_is_zero(self) -> None:
        """Odd-order moments of a zero-mean Gaussian are zero."""
        assert _gaussian_moments(variance=1.0, order=1) == 0.0
        assert _gaussian_moments(variance=1.0, order=3) == 0.0
        assert _gaussian_moments(variance=2.5, order=5) == 0.0

    def test_second_moment(self) -> None:
        """E[G^2] = variance."""
        assert _gaussian_moments(variance=2.0, order=2) == pytest.approx(2.0)

    def test_fourth_moment(self) -> None:
        """E[G^4] = 3 * variance^2."""
        var = 1.5
        assert _gaussian_moments(variance=var, order=4) == pytest.approx(3.0 * var**2)

    def test_zeroth_moment(self) -> None:
        """E[G^0] = 1."""
        assert _gaussian_moments(variance=1.0, order=0) == pytest.approx(1.0)

    def test_negative_order_raises(self) -> None:
        """Negative order raises ValueError."""
        with pytest.raises(ValueError, match="order must be non-negative"):
            _gaussian_moments(variance=1.0, order=-1)

    def test_negative_variance_raises(self) -> None:
        """Negative variance raises ValueError."""
        with pytest.raises(ValueError, match="variance must be non-negative"):
            _gaussian_moments(variance=-1.0, order=2)


# ── TestDiscreteConvolution ──────────────────────────────────────────────


class TestDiscreteConvolution:
    """Tests for the _discrete_convolution helper."""

    def test_single_coefficient(self) -> None:
        """[a] convolves to [a^2]."""
        result = _discrete_convolution(coeffs=(3.0,))
        assert result == pytest.approx([9.0])

    def test_two_coefficients(self) -> None:
        """[a, b] convolves to [a^2, 2ab, b^2]."""
        result = _discrete_convolution(coeffs=(2.0, 3.0))
        assert result == pytest.approx([4.0, 12.0, 9.0])

    def test_output_length(self) -> None:
        """Convolution of M+1 coefficients gives 2M+1 output length."""
        coeffs = (1.0, 2.0, 3.0)  # M=2, length=3
        result = _discrete_convolution(coeffs=coeffs)
        assert len(result) == 5  # 2*2+1 = 5


# ── TestKernelFunction ───────────────────────────────────────────────────


class TestKernelFunction:
    """Tests for the kernel_function function."""

    def test_exponential_at_zero(self) -> None:
        """Exponential kernel at t=0 equals epsilon^{H-1/2}."""
        h = 0.5
        eps = 0.1
        result = kernel_function(
            t=np.array([0.0]),
            kernel_type=KernelType.EXPONENTIAL,
            hurst=h,
            epsilon=eps,
        )
        expected = eps ** (h - 0.5)  # eps^0 = 1
        assert result[0] == pytest.approx(expected)

    def test_exponential_decays(self) -> None:
        """Exponential kernel decays with time."""
        t = np.array([0.0, 0.5, 1.0, 2.0])
        result = kernel_function(
            t=t, kernel_type=KernelType.EXPONENTIAL, hurst=0.4, epsilon=0.1
        )
        assert np.all(np.diff(result) < 0)

    def test_exponential_positive(self) -> None:
        """Exponential kernel is always positive."""
        t = np.linspace(0, 5, 100)
        result = kernel_function(
            t=t, kernel_type=KernelType.EXPONENTIAL, hurst=0.3, epsilon=0.2
        )
        assert np.all(result > 0)

    def test_fractional_at_one(self) -> None:
        """Fractional kernel at t=1 equals 1^{H-1/2} = 1 for any H."""
        for h in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = kernel_function(
                t=np.array([1.0]),
                kernel_type=KernelType.FRACTIONAL,
                hurst=h,
            )
            assert result[0] == pytest.approx(1.0)

    def test_fractional_at_zero(self) -> None:
        """Fractional kernel at t=0 is 0 (singular for H<0.5)."""
        result = kernel_function(
            t=np.array([0.0]),
            kernel_type=KernelType.FRACTIONAL,
            hurst=0.3,
        )
        assert result[0] == 0.0

    def test_shifted_fractional_positive(self) -> None:
        """Shifted fractional kernel is positive even at t=0."""
        result = kernel_function(
            t=np.array([0.0]),
            kernel_type=KernelType.SHIFTED_FRACTIONAL,
            hurst=0.3,
            epsilon=0.1,
        )
        assert result[0] > 0

    def test_log_modulated_positive_for_small_t(self) -> None:
        """Log-modulated kernel is positive for small t > 0."""
        result = kernel_function(
            t=np.array([0.01, 0.1, 0.5]),
            kernel_type=KernelType.LOG_MODULATED,
            hurst=0.3,
            theta=1.0,
            beta=0.5,
        )
        assert np.all(result > 0)

    def test_invalid_hurst_raises(self) -> None:
        """Hurst out of (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="hurst must be in"):
            kernel_function(
                t=np.array([1.0]),
                kernel_type=KernelType.EXPONENTIAL,
                hurst=0.0,
            )

    def test_invalid_epsilon_raises(self) -> None:
        """Non-positive epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            kernel_function(
                t=np.array([1.0]),
                kernel_type=KernelType.EXPONENTIAL,
                hurst=0.5,
                epsilon=0.0,
            )


# ── TestKernelVariance ───────────────────────────────────────────────────


class TestKernelVariance:
    """Tests for the kernel_variance function."""

    def test_exponential_closed_form(self) -> None:
        """Exponential kernel variance matches closed-form."""
        h = 0.4
        eps = 0.1
        upper = 1.0
        c = (0.5 - h) / eps
        two_h = 2.0 * (h - 0.5)
        expected = eps**two_h / (2.0 * c) * (1.0 - math.exp(-2.0 * c * upper))
        result = kernel_variance(
            upper_limit=upper,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=h,
            epsilon=eps,
        )
        assert result == pytest.approx(expected, rel=1e-10)

    def test_fractional_closed_form(self) -> None:
        """Fractional kernel variance matches closed-form T^{2H}/(2H)."""
        h = 0.3
        upper = 2.0
        expected = upper ** (2.0 * h) / (2.0 * h)
        result = kernel_variance(
            upper_limit=upper,
            kernel_type=KernelType.FRACTIONAL,
            hurst=h,
        )
        assert result == pytest.approx(expected, rel=1e-10)

    def test_zero_upper_limit(self) -> None:
        """Variance at T=0 is zero."""
        result = kernel_variance(
            upper_limit=0.0,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
        )
        assert result == 0.0

    def test_monotone_increasing(self) -> None:
        """Variance is monotonically increasing with T."""
        vals = [
            kernel_variance(
                upper_limit=t,
                kernel_type=KernelType.EXPONENTIAL,
                hurst=0.4,
                epsilon=0.1,
            )
            for t in [0.5, 1.0, 1.5, 2.0]
        ]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_negative_upper_limit_raises(self) -> None:
        """Negative upper limit raises ValueError."""
        with pytest.raises(ValueError, match="upper_limit must be non-negative"):
            kernel_variance(
                upper_limit=-1.0,
                kernel_type=KernelType.EXPONENTIAL,
                hurst=0.5,
            )

    def test_shifted_fractional_positive(self) -> None:
        """Shifted fractional variance is positive for T > 0."""
        result = kernel_variance(
            upper_limit=1.0,
            kernel_type=KernelType.SHIFTED_FRACTIONAL,
            hurst=0.3,
            epsilon=0.1,
        )
        assert result > 0


# ── TestPolynomialVolatility ─────────────────────────────────────────────


class TestPolynomialVolatility:
    """Tests for the polynomial_volatility function."""

    def test_constant_polynomial(self) -> None:
        """Constant polynomial p(x) = a0."""
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        result = polynomial_volatility(x=x, poly_coeffs=(3.0,))
        assert result == pytest.approx([3.0, 3.0, 3.0, 3.0])

    def test_linear_polynomial(self) -> None:
        """Linear polynomial p(x) = a0 + a1*x."""
        x = np.array([0.0, 1.0, 2.0])
        result = polynomial_volatility(x=x, poly_coeffs=(1.0, 2.0))
        assert result == pytest.approx([1.0, 3.0, 5.0])

    def test_quadratic_polynomial(self) -> None:
        """Quadratic polynomial p(x) = a0 + a1*x + a2*x^2."""
        x = np.array([0.0, 1.0, -1.0])
        result = polynomial_volatility(x=x, poly_coeffs=(1.0, 0.0, 1.0))
        assert result == pytest.approx([1.0, 2.0, 2.0])

    def test_empty_coeffs_raises(self) -> None:
        """Empty coefficients raise ValueError."""
        with pytest.raises(ValueError, match="poly_coeffs must be non-empty"):
            polynomial_volatility(x=np.array([1.0]), poly_coeffs=())

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        x = np.zeros(10)
        result = polynomial_volatility(x=x, poly_coeffs=(1.0, 2.0, 3.0))
        assert result.shape == (10,)


# ── TestNormalizationFactor ──────────────────────────────────────────────


class TestNormalizationFactor:
    """Tests for the _normalization_factor helper."""

    def test_constant_poly(self) -> None:
        """Constant polynomial gives g = a0^2."""
        result = _normalization_factor(poly_coeffs=(3.0,), kernel_variance_value=1.0)
        assert result == pytest.approx(9.0)

    def test_linear_poly(self) -> None:
        """Linear polynomial gives g = a0^2 + a1^2 * var."""
        var = 2.0
        a0, a1 = 1.0, 2.0
        # Convolution of [a0, a1] = [a0^2, 2*a0*a1, a1^2]
        # g = a0^2 * E[G^0] + 2*a0*a1 * E[G^1] + a1^2 * E[G^2]
        # g = a0^2 + 0 + a1^2 * var
        expected = a0**2 + a1**2 * var
        result = _normalization_factor(poly_coeffs=(a0, a1), kernel_variance_value=var)
        assert result == pytest.approx(expected)

    def test_always_positive_for_nonzero_coeffs(self) -> None:
        """Normalization factor is positive for non-zero constant term."""
        result = _normalization_factor(
            poly_coeffs=(0.5, 1.0, 0.3),
            kernel_variance_value=0.5,
        )
        assert result > 0


# ── TestForwardVariance ──────────────────────────────────────────────────


class TestForwardVariance:
    """Tests for the forward_variance function."""

    def test_at_time_zero_returns_xi_0(self) -> None:
        """Forward variance at t=0 matches xi_0."""
        config = _make_config()
        result = forward_variance(config=config, conditioning_time=0.0)
        assert isinstance(result, ForwardVarianceCurve)
        assert result.values == pytest.approx(config.xi_0)

    def test_positive_values(self) -> None:
        """Forward variance values are non-negative."""
        config = _make_config()
        result = forward_variance(
            config=config,
            conditioning_time=0.5,
            gaussian_values=np.array([0.0, 0.1, -0.1]),
        )
        assert np.all(result.values >= 0)

    def test_shape_matches_maturities(self) -> None:
        """Output shape matches number of maturities."""
        config = _make_config()
        result = forward_variance(config=config, conditioning_time=0.0)
        assert result.values.shape == config.maturities.shape

    def test_conditioning_time_stored(self) -> None:
        """Conditioning time is stored in result."""
        config = _make_config()
        result = forward_variance(config=config, conditioning_time=0.25)
        assert result.conditioning_time == 0.25


# ── TestSimulateGaussianProcess ──────────────────────────────────────────


class TestSimulateGaussianProcess:
    """Tests for the simulate_gaussian_process function."""

    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Output shape is (n_steps, n_sims)."""
        time_grid = np.linspace(0, 1, 11)
        result = simulate_gaussian_process(
            time_grid=time_grid,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            num_simulations=100,
            rng=rng,
        )
        assert result.shape == (11, 100)

    def test_starts_at_zero(self, rng: np.random.Generator) -> None:
        """Process starts at zero."""
        time_grid = np.linspace(0, 1, 11)
        result = simulate_gaussian_process(
            time_grid=time_grid,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            num_simulations=100,
            rng=rng,
        )
        assert np.all(result[0] == 0.0)

    def test_deterministic_seed(self) -> None:
        """Same seed produces identical paths."""
        time_grid = np.linspace(0, 1, 11)
        r1 = simulate_gaussian_process(
            time_grid=time_grid,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            num_simulations=50,
            rng=np.random.default_rng(42),
        )
        r2 = simulate_gaussian_process(
            time_grid=time_grid,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            num_simulations=50,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(r1, r2)

    def test_mean_near_zero(self) -> None:
        """Sample mean is near zero for large num_simulations."""
        time_grid = np.linspace(0, 1, 11)
        result = simulate_gaussian_process(
            time_grid=time_grid,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            epsilon=0.1,
            num_simulations=50_000,
            rng=np.random.default_rng(123),
        )
        # Mean at last time step should be near zero
        assert abs(np.mean(result[-1])) < 0.05

    def test_variance_matches_theory_exponential(self) -> None:
        """Variance at time T matches kernel_variance for exponential."""
        h = 0.4
        eps = 0.1
        upper = 1.0
        time_grid = np.linspace(0, upper, 51)
        result = simulate_gaussian_process(
            time_grid=time_grid,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=h,
            epsilon=eps,
            num_simulations=100_000,
            rng=np.random.default_rng(99),
        )
        sample_var = float(np.var(result[-1]))
        theory_var = kernel_variance(
            upper_limit=upper,
            kernel_type=KernelType.EXPONENTIAL,
            hurst=h,
            epsilon=eps,
        )
        assert sample_var == pytest.approx(theory_var, rel=0.1)

    def test_fractional_kernel_shape(self) -> None:
        """Fractional kernel simulation produces correct shape."""
        time_grid = np.linspace(0, 1, 6)
        result = simulate_gaussian_process(
            time_grid=time_grid,
            kernel_type=KernelType.FRACTIONAL,
            hurst=0.3,
            num_simulations=50,
            rng=np.random.default_rng(42),
        )
        assert result.shape == (6, 50)

    def test_invalid_hurst_raises(self) -> None:
        """Invalid hurst raises ValueError."""
        with pytest.raises(ValueError, match="hurst must be in"):
            simulate_gaussian_process(
                time_grid=np.linspace(0, 1, 5),
                kernel_type=KernelType.EXPONENTIAL,
                hurst=1.0,
                num_simulations=10,
            )

    def test_invalid_num_simulations_raises(self) -> None:
        """Non-positive num_simulations raises ValueError."""
        with pytest.raises(ValueError, match="num_simulations must be positive"):
            simulate_gaussian_process(
                time_grid=np.linspace(0, 1, 5),
                kernel_type=KernelType.EXPONENTIAL,
                hurst=0.5,
                num_simulations=0,
            )


# ── TestSimulatePaths ────────────────────────────────────────────────────


class TestSimulatePaths:
    """Tests for the simulate_paths function."""

    def test_output_shapes(self, rng: np.random.Generator) -> None:
        """All output arrays have correct shapes."""
        config = _make_config()
        result = simulate_paths(
            config=config,
            spot=100.0,
            time_horizon=1.0,
            num_steps=10,
            num_simulations=50,
            rng=rng,
        )
        assert isinstance(result, GPVSimulationResult)
        assert result.time_grid.shape == (11,)
        assert result.price_paths.shape == (11, 50)
        assert result.volatility_paths.shape == (11, 50)
        assert result.gaussian_paths.shape == (11, 50)

    def test_initial_price_equals_spot(self, rng: np.random.Generator) -> None:
        """Price at t=0 equals the spot price for all simulations."""
        config = _make_config()
        result = simulate_paths(
            config=config,
            spot=100.0,
            time_horizon=1.0,
            num_steps=10,
            num_simulations=50,
            rng=rng,
        )
        assert result.price_paths[0] == pytest.approx(np.full(50, 100.0), rel=1e-10)

    def test_all_prices_positive(self, rng: np.random.Generator) -> None:
        """All simulated prices are positive (log-space scheme)."""
        config = _make_config()
        result = simulate_paths(
            config=config,
            spot=100.0,
            time_horizon=1.0,
            num_steps=20,
            num_simulations=100,
            rng=rng,
        )
        assert np.all(result.price_paths > 0)

    def test_deterministic_seed(self) -> None:
        """Same seed produces identical results."""
        config = _make_config()
        r1 = simulate_paths(
            config=config,
            spot=100.0,
            time_horizon=1.0,
            num_steps=10,
            num_simulations=20,
            rng=np.random.default_rng(42),
        )
        r2 = simulate_paths(
            config=config,
            spot=100.0,
            time_horizon=1.0,
            num_steps=10,
            num_simulations=20,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(r1.price_paths, r2.price_paths)

    def test_positive_rate_increases_mean_price(self) -> None:
        """Positive risk-free rate increases mean terminal price."""
        config = _make_config()
        no_drift = simulate_paths(
            config=config,
            spot=100.0,
            time_horizon=1.0,
            num_steps=20,
            num_simulations=5000,
            risk_free_rate=0.0,
            rng=np.random.default_rng(42),
        )
        with_drift = simulate_paths(
            config=config,
            spot=100.0,
            time_horizon=1.0,
            num_steps=20,
            num_simulations=5000,
            risk_free_rate=0.10,
            rng=np.random.default_rng(42),
        )
        assert np.mean(with_drift.price_paths[-1]) > np.mean(no_drift.price_paths[-1])

    def test_invalid_spot_raises(self) -> None:
        """Non-positive spot raises ValueError."""
        config = _make_config()
        with pytest.raises(ValueError, match="spot must be positive"):
            simulate_paths(
                config=config,
                spot=0.0,
                time_horizon=1.0,
                rng=np.random.default_rng(42),
            )

    def test_invalid_time_horizon_raises(self) -> None:
        """Non-positive time_horizon raises ValueError."""
        config = _make_config()
        with pytest.raises(ValueError, match="time_horizon must be positive"):
            simulate_paths(
                config=config,
                spot=100.0,
                time_horizon=0.0,
                rng=np.random.default_rng(42),
            )


# ── TestComputeVIXSquared ────────────────────────────────────────────────


class TestComputeVIXSquared:
    """Tests for the compute_vix_squared function."""

    def test_positive(self) -> None:
        """VIX squared is positive."""
        config = _make_config()
        result = compute_vix_squared(config=config, observation_time=0.0)
        assert isinstance(result, VIXResult)
        assert float(result.vix_squared[0]) > 0

    def test_vix_is_sqrt_of_vix_squared(self) -> None:
        """VIX = sqrt(VIX^2)."""
        config = _make_config()
        result = compute_vix_squared(config=config, observation_time=0.0)
        assert float(result.vix_values[0]) == pytest.approx(
            math.sqrt(float(result.vix_squared[0]))
        )

    def test_flat_curve_gives_expected_vix(self) -> None:
        """Flat xi_0 curve gives VIX^2 approx 100^2 * xi_0."""
        xi_level = 0.04
        config = _make_config()
        result = compute_vix_squared(config=config, observation_time=0.0)
        expected_vix_sq = 100.0**2 * xi_level
        assert float(result.vix_squared[0]) == pytest.approx(expected_vix_sq, rel=0.05)

    def test_observation_time_stored(self) -> None:
        """Observation time is stored in result."""
        config = _make_config()
        result = compute_vix_squared(config=config, observation_time=0.5)
        assert result.observation_time == 0.5


# ── TestPriceEuropeanGPV ─────────────────────────────────────────────────


class TestPriceEuropeanGPV:
    """Tests for the price_european_gpv function."""

    def test_returns_correct_type(self) -> None:
        """Return type is GPVOptionPrice."""
        config = _make_config()
        result = price_european_gpv(
            config=config,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            num_steps=10,
            num_simulations=500,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, GPVOptionPrice)

    def test_call_non_negative(self) -> None:
        """Call price is non-negative."""
        config = _make_config()
        result = price_european_gpv(
            config=config,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            num_steps=10,
            num_simulations=1000,
            rng=np.random.default_rng(42),
        )
        assert result.call >= 0

    def test_put_non_negative(self) -> None:
        """Put price is non-negative."""
        config = _make_config()
        result = price_european_gpv(
            config=config,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            num_steps=10,
            num_simulations=1000,
            rng=np.random.default_rng(42),
        )
        assert result.put >= 0

    def test_standard_errors_positive(self) -> None:
        """Standard errors are positive."""
        config = _make_config()
        result = price_european_gpv(
            config=config,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            num_steps=10,
            num_simulations=1000,
            rng=np.random.default_rng(42),
        )
        assert result.call_std_error > 0
        assert result.put_std_error > 0

    def test_put_call_parity_within_mc_error(self) -> None:
        """Put-call parity holds within Monte Carlo error.

        C - P approx S - K*exp(-rT) for r=0 with constant vol.
        """
        config = _make_config()
        r = 0.0
        result = price_european_gpv(
            config=config,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=r,
            num_steps=50,
            num_simulations=20_000,
            rng=np.random.default_rng(42),
        )
        lhs = result.call - result.put
        rhs = 100.0 - 100.0 * math.exp(-r * 1.0)
        # Allow generous tolerance for MC error
        tol = 3.0 * max(result.call_std_error, result.put_std_error) + 1.0
        assert abs(lhs - rhs) < tol

    def test_put_call_parity_with_nonzero_rate(self) -> None:
        """Put-call parity C - P approx S - K*exp(-rT) for r > 0."""
        config = _make_config()
        r = 0.05
        result = price_european_gpv(
            config=config,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=r,
            num_steps=50,
            num_simulations=20_000,
            rng=np.random.default_rng(42),
        )
        lhs = result.call - result.put
        rhs = 100.0 - 100.0 * math.exp(-r * 1.0)
        tol = 3.0 * max(result.call_std_error, result.put_std_error) + 1.0
        assert abs(lhs - rhs) < tol

    def test_num_simulations_stored(self) -> None:
        """num_simulations is stored in result."""
        config = _make_config()
        result = price_european_gpv(
            config=config,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            num_steps=10,
            num_simulations=500,
            rng=np.random.default_rng(42),
        )
        assert result.num_simulations == 500

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("spot", 0.0, "spot must be positive"),
            ("spot", -1.0, "spot must be positive"),
            ("strike", 0.0, "strike must be positive"),
            ("strike", -1.0, "strike must be positive"),
            ("time_to_expiry", 0.0, "time_to_expiry must be positive"),
            ("time_to_expiry", -1.0, "time_to_expiry must be positive"),
        ],
    )
    def test_invalid_inputs_raise(self, field: str, value: float, match: str) -> None:
        """Invalid inputs raise ValueError."""
        config = _make_config()
        defaults: dict[str, object] = {
            "config": config,
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.0,
            "num_steps": 5,
            "num_simulations": 10,
            "rng": np.random.default_rng(42),
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            price_european_gpv(**defaults)  # type: ignore[arg-type]


# ── TestGPVModelConfig Validation ────────────────────────────────────────


class TestGPVConfigValidation:
    """Tests for GPV config validation."""

    def test_invalid_hurst_raises(self) -> None:
        """Hurst out of (0, 1) raises ValueError."""
        config = GPVModelConfig(
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.0,
            poly_coeffs=(1.0,),
            xi_0=_FLAT_XI.copy(),
            maturities=_MATURITIES.copy(),
        )
        with pytest.raises(ValueError, match="hurst must be in"):
            forward_variance(config=config, conditioning_time=0.0)

    def test_empty_poly_coeffs_raises(self) -> None:
        """Empty poly_coeffs raises ValueError."""
        config = GPVModelConfig(
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            poly_coeffs=(),
            xi_0=_FLAT_XI.copy(),
            maturities=_MATURITIES.copy(),
        )
        with pytest.raises(ValueError, match="poly_coeffs must be non-empty"):
            forward_variance(config=config, conditioning_time=0.0)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched xi_0 and maturities raises ValueError."""
        config = GPVModelConfig(
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            poly_coeffs=(1.0,),
            xi_0=np.array([0.04, 0.04]),
            maturities=np.array([0.5]),
        )
        with pytest.raises(ValueError, match="xi_0 and maturities must have the same"):
            forward_variance(config=config, conditioning_time=0.0)

    def test_invalid_rho_raises(self) -> None:
        """rho out of [-1, 1] raises ValueError."""
        config = GPVModelConfig(
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            poly_coeffs=(1.0,),
            xi_0=_FLAT_XI.copy(),
            maturities=_MATURITIES.copy(),
            rho=1.5,
        )
        with pytest.raises(ValueError, match="rho must be in"):
            simulate_paths(
                config=config,
                spot=100.0,
                time_horizon=1.0,
                num_steps=5,
                num_simulations=10,
                rng=np.random.default_rng(42),
            )

    def test_invalid_epsilon_raises(self) -> None:
        """Non-positive epsilon raises ValueError."""
        config = GPVModelConfig(
            kernel_type=KernelType.EXPONENTIAL,
            hurst=0.5,
            poly_coeffs=(1.0,),
            xi_0=_FLAT_XI.copy(),
            maturities=_MATURITIES.copy(),
            epsilon=-0.1,
        )
        with pytest.raises(ValueError, match="epsilon must be positive"):
            forward_variance(config=config, conditioning_time=0.0)
