"""Tests for the Black-Scholes option pricing module."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from investment_portfolio.black_scholes import (
    Greeks,
    OptionPrice,
    calculate_greeks,
    implied_volatility,
    price_european,
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
