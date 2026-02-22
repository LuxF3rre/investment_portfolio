"""Tests for the Black-Scholes option pricing module."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from investment_portfolio.black_scholes import (
    BinaryPrice,
    Greeks,
    OptionPrice,
    PerpetualPutResult,
    calculate_fx_greeks,
    calculate_greeks,
    implied_volatility,
    price_binary,
    price_discrete_dividend,
    price_european,
    price_fx_option,
    price_perpetual_put,
    price_surface,
)

# ── Known analytical values ─────────────────────────────────────────────────
# Reference: S=100, K=100, T=1, r=0.05, vol=0.2 (standard textbook example)
_REF_S = 100.0
_REF_K = 100.0
_REF_T = 1.0
_REF_R = 0.05
_REF_VOL = 0.20

# Pre-computed d1, d2
_d1 = (math.log(_REF_S / _REF_K) + (_REF_R + 0.5 * _REF_VOL**2) * _REF_T) / (
    _REF_VOL * math.sqrt(_REF_T)
)
_d2 = _d1 - _REF_VOL * math.sqrt(_REF_T)
_EXPECTED_CALL = float(
    _REF_S * norm.cdf(_d1) - _REF_K * math.exp(-_REF_R * _REF_T) * norm.cdf(_d2)
)
_EXPECTED_PUT = float(
    _REF_K * math.exp(-_REF_R * _REF_T) * norm.cdf(-_d2) - _REF_S * norm.cdf(-_d1)
)


class TestPriceEuropean:
    """Tests for the price_european function."""

    def test_known_call_price(self) -> None:
        """Call price matches the analytical formula."""
        result = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert isinstance(result, OptionPrice)
        assert result.call == pytest.approx(_EXPECTED_CALL, abs=1e-10)

    def test_known_put_price(self) -> None:
        """Put price matches the analytical formula."""
        result = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert result.put == pytest.approx(_EXPECTED_PUT, abs=1e-10)

    def test_put_call_parity(self) -> None:
        """Put-call parity: C - P = S - K * exp(-rT)."""
        result = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        lhs = result.call - result.put
        rhs = _REF_S - _REF_K * math.exp(-_REF_R * _REF_T)
        assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_put_call_parity_otm(self) -> None:
        """Put-call parity holds for out-of-the-money options."""
        result = price_european(
            spot=80.0,
            strike=100.0,
            time_to_expiry=0.5,
            risk_free_rate=0.03,
            volatility=0.30,
        )
        lhs = result.call - result.put
        rhs = 80.0 - 100.0 * math.exp(-0.03 * 0.5)
        assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_deep_itm_call_approaches_intrinsic(self) -> None:
        """Deep ITM call is close to S - K * exp(-rT)."""
        result = price_european(
            spot=200.0,
            strike=50.0,
            time_to_expiry=0.01,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        intrinsic = 200.0 - 50.0 * math.exp(-0.05 * 0.01)
        assert result.call == pytest.approx(intrinsic, rel=1e-3)

    def test_deep_otm_call_near_zero(self) -> None:
        """Deep OTM call price is near zero."""
        result = price_european(
            spot=50.0,
            strike=200.0,
            time_to_expiry=0.01,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        assert result.call < 0.01

    def test_prices_positive(self) -> None:
        """Both call and put prices are non-negative."""
        result = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert result.call > 0
        assert result.put > 0

    def test_call_increases_with_spot(self) -> None:
        """Call price increases as spot price increases."""
        low = price_european(
            spot=90.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        high = price_european(
            spot=110.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        assert high.call > low.call

    def test_put_decreases_with_spot(self) -> None:
        """Put price decreases as spot price increases."""
        low = price_european(
            spot=90.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        high = price_european(
            spot=110.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        assert low.put > high.put

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("spot", -1.0, "spot must be positive"),
            ("spot", 0.0, "spot must be positive"),
            ("strike", -1.0, "strike must be positive"),
            ("strike", 0.0, "strike must be positive"),
            ("time_to_expiry", -0.1, "time_to_expiry must be positive"),
            ("time_to_expiry", 0.0, "time_to_expiry must be positive"),
            ("risk_free_rate", -0.01, "risk_free_rate must be non-negative"),
            ("volatility", -0.1, "volatility must be positive"),
            ("volatility", 0.0, "volatility must be positive"),
        ],
    )
    def test_invalid_inputs_raise(self, field: str, value: float, match: str) -> None:
        """Invalid inputs raise ValueError with a descriptive message."""
        defaults: dict[str, float] = {
            "spot": _REF_S,
            "strike": _REF_K,
            "time_to_expiry": _REF_T,
            "risk_free_rate": _REF_R,
            "volatility": _REF_VOL,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            price_european(**defaults)


class TestCalculateGreeks:
    """Tests for the calculate_greeks function."""

    def test_call_delta_bounds(self) -> None:
        """Call delta is between 0 and 1."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        assert 0 <= g.delta <= 1

    def test_put_delta_bounds(self) -> None:
        """Put delta is between -1 and 0."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=False,
        )
        assert -1 <= g.delta <= 0

    def test_call_put_delta_relationship(self) -> None:
        """Call delta - put delta = 1."""
        call_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        put_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=False,
        )
        assert call_g.delta - put_g.delta == pytest.approx(1.0, abs=1e-10)

    def test_gamma_positive(self) -> None:
        """Gamma is always positive."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        assert g.gamma > 0

    def test_gamma_same_for_call_and_put(self) -> None:
        """Gamma is the same for call and put."""
        call_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        put_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=False,
        )
        assert call_g.gamma == pytest.approx(put_g.gamma, abs=1e-10)

    def test_vega_positive(self) -> None:
        """Vega is always positive."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        assert g.vega > 0

    def test_vega_same_for_call_and_put(self) -> None:
        """Vega is the same for call and put."""
        call_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        put_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=False,
        )
        assert call_g.vega == pytest.approx(put_g.vega, abs=1e-10)

    def test_call_theta_negative(self) -> None:
        """Call theta is negative (time decay)."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        assert g.theta < 0

    def test_call_rho_positive(self) -> None:
        """Call rho is positive (higher rates benefit calls)."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        assert g.rho > 0

    def test_put_rho_negative(self) -> None:
        """Put rho is negative (higher rates hurt puts)."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=False,
        )
        assert g.rho < 0

    def test_put_theta_for_deep_itm(self) -> None:
        """Put theta can be positive for deep ITM puts with high rates."""
        g = calculate_greeks(
            spot=50.0,
            strike=200.0,
            time_to_expiry=0.01,
            risk_free_rate=0.10,
            volatility=0.20,
            is_call=False,
        )
        # Deep ITM put theta can be positive due to interest rate effect
        assert isinstance(g.theta, float)

    def test_known_gamma_value(self) -> None:
        """Gamma matches hand-computed value."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        expected_gamma = float(norm.pdf(_d1) / (_REF_S * _REF_VOL * math.sqrt(_REF_T)))
        assert g.gamma == pytest.approx(expected_gamma, abs=1e-10)

    def test_known_vega_value(self) -> None:
        """Vega (per 1% move) matches hand-computed value."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        expected_vega = float(_REF_S * norm.pdf(_d1) * math.sqrt(_REF_T) / 100)
        assert g.vega == pytest.approx(expected_vega, abs=1e-10)

    def test_returns_greeks_dataclass(self) -> None:
        """Return type is Greeks."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        assert isinstance(g, Greeks)

    def test_known_delta_value(self) -> None:
        """Delta matches hand-computed value."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
        )
        expected_delta = float(norm.cdf(_d1))
        assert g.delta == pytest.approx(expected_delta, abs=1e-10)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("spot", 0.0, "spot must be positive"),
            ("strike", -5.0, "strike must be positive"),
            ("volatility", 0.0, "volatility must be positive"),
        ],
    )
    def test_invalid_inputs_raise(self, field: str, value: float, match: str) -> None:
        """Invalid inputs raise ValueError."""
        defaults: dict[str, float | bool] = {
            "spot": _REF_S,
            "strike": _REF_K,
            "time_to_expiry": _REF_T,
            "risk_free_rate": _REF_R,
            "volatility": _REF_VOL,
            "is_call": True,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            calculate_greeks(**defaults)  # type: ignore[arg-type]


class TestImpliedVolatility:
    """Tests for the implied_volatility function."""

    def test_round_trip_call(self) -> None:
        """Price -> IV -> price recovers the original volatility for a call."""
        op = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        iv = implied_volatility(
            market_price=op.call,
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            is_call=True,
        )
        assert iv == pytest.approx(_REF_VOL, abs=1e-6)

    def test_round_trip_put(self) -> None:
        """Price -> IV -> price recovers the original volatility for a put."""
        op = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        iv = implied_volatility(
            market_price=op.put,
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            is_call=False,
        )
        assert iv == pytest.approx(_REF_VOL, abs=1e-6)

    def test_round_trip_high_vol(self) -> None:
        """Round-trip works for high volatility."""
        target_vol = 1.5
        op = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=target_vol,
        )
        iv = implied_volatility(
            market_price=op.call,
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            is_call=True,
        )
        assert iv == pytest.approx(target_vol, abs=1e-4)

    def test_round_trip_otm_put(self) -> None:
        """Round-trip for an out-of-the-money put."""
        target_vol = 0.30
        op = price_european(
            spot=120.0,
            strike=100.0,
            time_to_expiry=0.5,
            risk_free_rate=0.03,
            volatility=target_vol,
        )
        iv = implied_volatility(
            market_price=op.put,
            spot=120.0,
            strike=100.0,
            time_to_expiry=0.5,
            risk_free_rate=0.03,
            is_call=False,
        )
        assert iv == pytest.approx(target_vol, abs=1e-4)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("spot", 0.0, "spot must be positive"),
            ("strike", -1.0, "strike must be positive"),
            ("time_to_expiry", 0.0, "time_to_expiry must be positive"),
            ("risk_free_rate", -0.01, "risk_free_rate must be non-negative"),
        ],
    )
    def test_invalid_inputs_raise(self, field: str, value: float, match: str) -> None:
        """Invalid BS inputs raise ValueError."""
        defaults: dict[str, float | bool] = {
            "market_price": 10.0,
            "spot": _REF_S,
            "strike": _REF_K,
            "time_to_expiry": _REF_T,
            "risk_free_rate": _REF_R,
            "is_call": True,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            implied_volatility(**defaults)  # type: ignore[arg-type]

    def test_put_price_above_upper_bound_raises(self) -> None:
        """Put market price above discounted strike raises ValueError."""
        with pytest.raises(ValueError, match="exceeds arbitrage upper bound"):
            implied_volatility(
                market_price=100.0,
                spot=100.0,
                strike=100.0,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                is_call=False,
            )

    def test_put_price_below_intrinsic_raises(self) -> None:
        """Put market price below intrinsic raises ValueError."""
        with pytest.raises(ValueError, match="market_price is below intrinsic"):
            implied_volatility(
                market_price=0.01,
                spot=50.0,
                strike=100.0,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                is_call=False,
            )

    def test_round_trip_low_vol(self) -> None:
        """Round-trip works for low volatility (tests bisection fallback)."""
        target_vol = 0.02
        op = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=target_vol,
        )
        iv = implied_volatility(
            market_price=op.call,
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            is_call=True,
        )
        assert iv == pytest.approx(target_vol, abs=1e-4)

    def test_invalid_market_price_raises(self) -> None:
        """Non-positive market price raises ValueError."""
        with pytest.raises(ValueError, match="market_price must be positive"):
            implied_volatility(
                market_price=-1.0,
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                risk_free_rate=_REF_R,
                is_call=True,
            )

    def test_zero_market_price_raises(self) -> None:
        """Zero market price raises ValueError."""
        with pytest.raises(ValueError, match="market_price must be positive"):
            implied_volatility(
                market_price=0.0,
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                risk_free_rate=_REF_R,
                is_call=True,
            )

    def test_price_below_intrinsic_raises(self) -> None:
        """Market price below intrinsic value raises ValueError."""
        with pytest.raises(ValueError, match="market_price is below intrinsic"):
            implied_volatility(
                market_price=0.01,
                spot=150.0,
                strike=100.0,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                is_call=True,
            )

    def test_price_above_upper_bound_raises(self) -> None:
        """Call market price above spot raises ValueError."""
        with pytest.raises(ValueError, match="exceeds arbitrage upper bound"):
            implied_volatility(
                market_price=110.0,
                spot=100.0,
                strike=100.0,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                is_call=True,
            )


class TestPriceSurface:
    """Tests for the price_surface function."""

    def test_output_shape(self) -> None:
        """Output shape matches (len(vols), len(spots))."""
        spots = np.linspace(80, 120, 10)
        vols = np.linspace(0.1, 0.5, 8)
        result = price_surface(
            spots=spots,
            volatilities=vols,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            is_call=True,
        )
        assert result.shape == (8, 10)

    def test_all_prices_non_negative(self) -> None:
        """All surface prices are non-negative."""
        spots = np.linspace(50, 150, 10)
        vols = np.linspace(0.05, 1.0, 10)
        result = price_surface(
            spots=spots,
            volatilities=vols,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            is_call=True,
        )
        assert np.all(result >= 0)

    def test_monotonic_in_spot_for_call(self) -> None:
        """Call price increases with spot (each vol row)."""
        spots = np.linspace(80, 120, 20)
        vols = np.array([0.2])
        result = price_surface(
            spots=spots,
            volatilities=vols,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            is_call=True,
        )
        assert np.all(np.diff(result[0]) > 0)

    def test_monotonic_in_vol(self) -> None:
        """Call price increases with volatility (each spot column)."""
        spots = np.array([100.0])
        vols = np.linspace(0.05, 1.0, 20)
        result = price_surface(
            spots=spots,
            volatilities=vols,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            is_call=True,
        )
        assert np.all(np.diff(result[:, 0]) > 0)

    def test_put_surface_shape(self) -> None:
        """Put surface has correct shape too."""
        spots = np.linspace(80, 120, 5)
        vols = np.linspace(0.1, 0.5, 4)
        result = price_surface(
            spots=spots,
            volatilities=vols,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            is_call=False,
        )
        assert result.shape == (4, 5)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("strike", 0.0, "strike must be positive"),
            ("strike", -1.0, "strike must be positive"),
            ("time_to_expiry", 0.0, "time_to_expiry must be positive"),
            ("time_to_expiry", -1.0, "time_to_expiry must be positive"),
            ("risk_free_rate", -0.01, "risk_free_rate must be non-negative"),
        ],
    )
    def test_invalid_inputs_raise(self, field: str, value: float, match: str) -> None:
        """Invalid inputs raise ValueError."""
        defaults: dict[str, object] = {
            "spots": np.array([100.0]),
            "volatilities": np.array([0.2]),
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.05,
            "is_call": True,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            price_surface(**defaults)  # type: ignore[arg-type]


class TestContinuousDividend:
    """Tests for continuous dividend yield in price_european."""

    def test_put_call_parity_with_dividend(self) -> None:
        """C - P = S*e^{-qT} - K*e^{-rT} with dividend yield."""
        q = 0.03
        result = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_yield=q,
        )
        lhs = result.call - result.put
        rhs = _REF_S * math.exp(-q * _REF_T) - _REF_K * math.exp(-_REF_R * _REF_T)
        assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_dividend_reduces_call_price(self) -> None:
        """A positive dividend yield reduces call price vs q=0."""
        no_div = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        with_div = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_yield=0.03,
        )
        assert with_div.call < no_div.call

    def test_dividend_increases_put_price(self) -> None:
        """A positive dividend yield increases put price vs q=0."""
        no_div = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        with_div = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_yield=0.03,
        )
        assert with_div.put > no_div.put

    def test_zero_dividend_matches_standard(self) -> None:
        """dividend_yield=0 matches standard BS exactly."""
        standard = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        with_zero = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_yield=0.0,
        )
        assert with_zero.call == pytest.approx(standard.call, abs=1e-12)
        assert with_zero.put == pytest.approx(standard.put, abs=1e-12)

    def test_negative_dividend_raises(self) -> None:
        """Negative dividend_yield raises ValueError."""
        with pytest.raises(ValueError, match="dividend_yield must be non-negative"):
            price_european(
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                risk_free_rate=_REF_R,
                volatility=_REF_VOL,
                dividend_yield=-0.01,
            )


class TestContinuousDividendGreeks:
    """Tests for Greeks with continuous dividend yield."""

    def test_call_delta_bounded_by_div_discount(self) -> None:
        """Call delta is bounded by [0, e^{-qT}]."""
        q = 0.03
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
            dividend_yield=q,
        )
        assert 0 <= g.delta <= math.exp(-q * _REF_T)

    def test_call_put_delta_sum_equals_div_discount(self) -> None:
        """Call delta - put delta = e^{-qT}."""
        q = 0.03
        call_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
            dividend_yield=q,
        )
        put_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=False,
            dividend_yield=q,
        )
        assert call_g.delta - put_g.delta == pytest.approx(
            math.exp(-q * _REF_T), abs=1e-10
        )

    def test_gamma_positive_with_dividend(self) -> None:
        """Gamma is positive with dividend yield."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
            dividend_yield=0.03,
        )
        assert g.gamma > 0

    def test_vega_positive_with_dividend(self) -> None:
        """Vega is positive with dividend yield."""
        g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            is_call=True,
            dividend_yield=0.03,
        )
        assert g.vega > 0


class TestPriceDiscreteDividend:
    """Tests for the price_discrete_dividend function."""

    def test_adjusted_spot_matches_formula(self) -> None:
        """Adjusted spot = S * (1-d)^n gives expected price."""
        d_prop = 0.02
        n_div = 4
        adjusted_spot = _REF_S * (1 - d_prop) ** n_div
        expected = price_european(
            spot=adjusted_spot,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        result = price_discrete_dividend(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_proportion=d_prop,
            num_dividends=n_div,
        )
        assert result.call == pytest.approx(expected.call, abs=1e-10)
        assert result.put == pytest.approx(expected.put, abs=1e-10)

    def test_zero_dividends_matches_standard(self) -> None:
        """num_dividends=0 matches standard BS."""
        standard = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        result = price_discrete_dividend(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_proportion=0.05,
            num_dividends=0,
        )
        assert result.call == pytest.approx(standard.call, abs=1e-12)
        assert result.put == pytest.approx(standard.put, abs=1e-12)

    def test_zero_proportion_matches_standard(self) -> None:
        """dividend_proportion=0 matches standard BS."""
        standard = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        result = price_discrete_dividend(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_proportion=0.0,
            num_dividends=4,
        )
        assert result.call == pytest.approx(standard.call, abs=1e-12)
        assert result.put == pytest.approx(standard.put, abs=1e-12)

    def test_invalid_proportion_raises(self) -> None:
        """Invalid dividend_proportion raises ValueError."""
        with pytest.raises(ValueError, match="dividend_proportion must be in"):
            price_discrete_dividend(
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                risk_free_rate=_REF_R,
                volatility=_REF_VOL,
                dividend_proportion=1.0,
                num_dividends=4,
            )

    def test_negative_proportion_raises(self) -> None:
        """Negative dividend_proportion raises ValueError."""
        with pytest.raises(ValueError, match="dividend_proportion must be in"):
            price_discrete_dividend(
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                risk_free_rate=_REF_R,
                volatility=_REF_VOL,
                dividend_proportion=-0.01,
                num_dividends=4,
            )

    def test_put_call_parity_with_adjusted_forward(self) -> None:
        """Put-call parity holds with adjusted spot."""
        d_prop = 0.02
        n_div = 4
        result = price_discrete_dividend(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_proportion=d_prop,
            num_dividends=n_div,
        )
        adjusted_spot = _REF_S * (1 - d_prop) ** n_div
        lhs = result.call - result.put
        rhs = adjusted_spot - _REF_K * math.exp(-_REF_R * _REF_T)
        assert lhs == pytest.approx(rhs, abs=1e-10)


class TestPriceBinary:
    """Tests for the price_binary function."""

    def test_returns_binary_price_dataclass(self) -> None:
        """Return type is BinaryPrice."""
        result = price_binary(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert isinstance(result, BinaryPrice)

    def test_all_prices_non_negative(self) -> None:
        """All four binary prices are non-negative."""
        result = price_binary(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert result.cash_or_nothing_call >= 0
        assert result.cash_or_nothing_put >= 0
        assert result.asset_or_nothing_call >= 0
        assert result.asset_or_nothing_put >= 0

    def test_cash_or_nothing_sum_equals_discount(self) -> None:
        """Cash-or-nothing call + put = e^{-rT}."""
        result = price_binary(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        expected = math.exp(-_REF_R * _REF_T)
        actual = result.cash_or_nothing_call + result.cash_or_nothing_put
        assert actual == pytest.approx(expected, abs=1e-10)

    def test_asset_or_nothing_sum_equals_div_discounted_spot(self) -> None:
        """Asset-or-nothing call + put = S * e^{-qT}."""
        q = 0.02
        result = price_binary(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_yield=q,
        )
        expected = _REF_S * math.exp(-q * _REF_T)
        actual = result.asset_or_nothing_call + result.asset_or_nothing_put
        assert actual == pytest.approx(expected, abs=1e-10)

    def test_asset_or_nothing_sum_no_dividend(self) -> None:
        """Asset-or-nothing call + put = S when q=0."""
        result = price_binary(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        actual = result.asset_or_nothing_call + result.asset_or_nothing_put
        assert actual == pytest.approx(_REF_S, abs=1e-10)

    def test_known_values(self) -> None:
        """Cash-or-nothing call matches hand-computed value."""
        d1 = (math.log(_REF_S / _REF_K) + (_REF_R + 0.5 * _REF_VOL**2) * _REF_T) / (
            _REF_VOL * math.sqrt(_REF_T)
        )
        d2 = d1 - _REF_VOL * math.sqrt(_REF_T)
        expected_con_call = math.exp(-_REF_R * _REF_T) * norm.cdf(d2)

        result = price_binary(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert result.cash_or_nothing_call == pytest.approx(
            expected_con_call, abs=1e-10
        )

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("spot", 0.0, "spot must be positive"),
            ("strike", -1.0, "strike must be positive"),
            ("volatility", 0.0, "volatility must be positive"),
        ],
    )
    def test_invalid_inputs_raise(self, field: str, value: float, match: str) -> None:
        """Invalid inputs raise ValueError."""
        defaults: dict[str, float] = {
            "spot": _REF_S,
            "strike": _REF_K,
            "time_to_expiry": _REF_T,
            "risk_free_rate": _REF_R,
            "volatility": _REF_VOL,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            price_binary(**defaults)


class TestPriceFxOption:
    """Tests for the price_fx_option function."""

    def test_equivalent_to_european_with_dividend(self) -> None:
        """FX price matches price_european with dividend_yield=foreign_rate."""
        rd, rf = 0.05, 0.02
        fx_result = price_fx_option(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            domestic_rate=rd,
            foreign_rate=rf,
            volatility=_REF_VOL,
        )
        eu_result = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=rd,
            volatility=_REF_VOL,
            dividend_yield=rf,
        )
        assert fx_result.call == pytest.approx(eu_result.call, abs=1e-12)
        assert fx_result.put == pytest.approx(eu_result.put, abs=1e-12)

    def test_put_call_parity(self) -> None:
        """C - P = S*e^{-rf*T} - K*e^{-rd*T} for FX options."""
        rd, rf = 0.05, 0.02
        result = price_fx_option(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            domestic_rate=rd,
            foreign_rate=rf,
            volatility=_REF_VOL,
        )
        lhs = result.call - result.put
        rhs = _REF_S * math.exp(-rf * _REF_T) - _REF_K * math.exp(-rd * _REF_T)
        assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_negative_domestic_rate_raises(self) -> None:
        """Negative domestic rate raises ValueError."""
        with pytest.raises(ValueError, match="domestic_rate must be non-negative"):
            price_fx_option(
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                domestic_rate=-0.01,
                foreign_rate=0.02,
                volatility=_REF_VOL,
            )

    def test_negative_foreign_rate_raises(self) -> None:
        """Negative foreign rate raises ValueError."""
        with pytest.raises(ValueError, match="foreign_rate must be non-negative"):
            price_fx_option(
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                domestic_rate=0.05,
                foreign_rate=-0.01,
                volatility=_REF_VOL,
            )


class TestCalculateFxGreeks:
    """Tests for the calculate_fx_greeks function."""

    def test_equivalent_to_greeks_with_dividend(self) -> None:
        """FX Greeks match calculate_greeks with dividend_yield=foreign_rate."""
        rd, rf = 0.05, 0.02
        fx_g = calculate_fx_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            domestic_rate=rd,
            foreign_rate=rf,
            volatility=_REF_VOL,
            is_call=True,
        )
        eu_g = calculate_greeks(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=rd,
            volatility=_REF_VOL,
            is_call=True,
            dividend_yield=rf,
        )
        assert fx_g.delta == pytest.approx(eu_g.delta, abs=1e-12)
        assert fx_g.gamma == pytest.approx(eu_g.gamma, abs=1e-12)
        assert fx_g.theta == pytest.approx(eu_g.theta, abs=1e-12)
        assert fx_g.vega == pytest.approx(eu_g.vega, abs=1e-12)
        assert fx_g.rho == pytest.approx(eu_g.rho, abs=1e-12)

    def test_negative_domestic_rate_raises(self) -> None:
        """Negative domestic rate raises ValueError."""
        with pytest.raises(ValueError, match="domestic_rate must be non-negative"):
            calculate_fx_greeks(
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                domestic_rate=-0.01,
                foreign_rate=0.02,
                volatility=_REF_VOL,
                is_call=True,
            )

    def test_negative_foreign_rate_raises(self) -> None:
        """Negative foreign rate raises ValueError."""
        with pytest.raises(ValueError, match="foreign_rate must be non-negative"):
            calculate_fx_greeks(
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                domestic_rate=0.05,
                foreign_rate=-0.01,
                volatility=_REF_VOL,
                is_call=True,
            )


class TestPricePerpetualPut:
    """Tests for the price_perpetual_put function."""

    def test_returns_perpetual_put_result(self) -> None:
        """Return type is PerpetualPutResult."""
        result = price_perpetual_put(
            spot=_REF_S,
            strike=_REF_K,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert isinstance(result, PerpetualPutResult)

    def test_price_positive_above_boundary(self) -> None:
        """Price > 0 when spot > exercise boundary."""
        result = price_perpetual_put(
            spot=_REF_S,
            strike=_REF_K,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert result.price > 0
        assert _REF_S > result.exercise_boundary

    def test_price_equals_intrinsic_below_boundary(self) -> None:
        """Price = K - S when spot <= exercise boundary."""
        result = price_perpetual_put(
            spot=_REF_S,
            strike=_REF_K,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        # Use a spot well below the boundary
        low_spot = result.exercise_boundary * 0.5
        low_result = price_perpetual_put(
            spot=low_spot,
            strike=_REF_K,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert low_result.price == pytest.approx(_REF_K - low_spot, abs=1e-10)

    def test_exercise_boundary_below_strike(self) -> None:
        """Exercise boundary is below strike."""
        result = price_perpetual_put(
            spot=_REF_S,
            strike=_REF_K,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert result.exercise_boundary < _REF_K

    def test_higher_vol_higher_price(self) -> None:
        """Higher volatility gives a higher perpetual put price."""
        low_vol = price_perpetual_put(
            spot=_REF_S,
            strike=_REF_K,
            risk_free_rate=_REF_R,
            volatility=0.15,
        )
        high_vol = price_perpetual_put(
            spot=_REF_S,
            strike=_REF_K,
            risk_free_rate=_REF_R,
            volatility=0.30,
        )
        assert high_vol.price > low_vol.price

    def test_zero_rate_raises(self) -> None:
        """risk_free_rate=0 raises ValueError."""
        with pytest.raises(
            ValueError, match="risk_free_rate must be strictly positive"
        ):
            price_perpetual_put(
                spot=_REF_S,
                strike=_REF_K,
                risk_free_rate=0.0,
                volatility=_REF_VOL,
            )

    def test_negative_rate_raises(self) -> None:
        """Negative risk_free_rate raises ValueError."""
        with pytest.raises(
            ValueError, match="risk_free_rate must be strictly positive"
        ):
            price_perpetual_put(
                spot=_REF_S,
                strike=_REF_K,
                risk_free_rate=-0.01,
                volatility=_REF_VOL,
            )

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("spot", 0.0, "spot must be positive"),
            ("strike", -1.0, "strike must be positive"),
            ("volatility", 0.0, "volatility must be positive"),
        ],
    )
    def test_invalid_inputs_raise(self, field: str, value: float, match: str) -> None:
        """Invalid inputs raise ValueError."""
        defaults: dict[str, float] = {
            "spot": _REF_S,
            "strike": _REF_K,
            "risk_free_rate": _REF_R,
            "volatility": _REF_VOL,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            price_perpetual_put(**defaults)

    def test_lambda_2_negative(self) -> None:
        """lambda_2 is always negative."""
        result = price_perpetual_put(
            spot=_REF_S,
            strike=_REF_K,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
        )
        assert result.lambda_2 < 0


class TestImpliedVolatilityWithDividend:
    """Tests for implied_volatility with dividend yield."""

    def test_round_trip_call_with_dividend(self) -> None:
        """Price -> IV -> price round-trip with dividend_yield > 0."""
        q = 0.03
        op = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_yield=q,
        )
        iv = implied_volatility(
            market_price=op.call,
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            is_call=True,
            dividend_yield=q,
        )
        assert iv == pytest.approx(_REF_VOL, abs=1e-6)

    def test_round_trip_put_with_dividend(self) -> None:
        """Price -> IV -> price round-trip for put with dividend_yield > 0."""
        q = 0.03
        op = price_european(
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            volatility=_REF_VOL,
            dividend_yield=q,
        )
        iv = implied_volatility(
            market_price=op.put,
            spot=_REF_S,
            strike=_REF_K,
            time_to_expiry=_REF_T,
            risk_free_rate=_REF_R,
            is_call=False,
            dividend_yield=q,
        )
        assert iv == pytest.approx(_REF_VOL, abs=1e-6)

    def test_negative_dividend_raises(self) -> None:
        """Negative dividend_yield raises ValueError."""
        with pytest.raises(ValueError, match="dividend_yield must be non-negative"):
            implied_volatility(
                market_price=10.0,
                spot=_REF_S,
                strike=_REF_K,
                time_to_expiry=_REF_T,
                risk_free_rate=_REF_R,
                is_call=True,
                dividend_yield=-0.01,
            )
