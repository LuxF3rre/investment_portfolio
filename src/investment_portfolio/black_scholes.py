"""Black-Scholes-Merton European option pricing and extensions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = [
    "BinaryPrice",
    "Greeks",
    "OptionPrice",
    "PerpetualPutResult",
    "calculate_fx_greeks",
    "calculate_greeks",
    "implied_volatility",
    "price_binary",
    "price_discrete_dividend",
    "price_european",
    "price_fx_option",
    "price_perpetual_put",
    "price_surface",
]


@dataclass(frozen=True, slots=True)
class OptionPrice:
    """European call and put prices."""

    call: float
    put: float


@dataclass(frozen=True, slots=True)
class Greeks:
    """Option sensitivity measures."""

    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float  # per 1 % vol move
    rho: float  # per 1 % rate move


@dataclass(frozen=True, slots=True)
class BinaryPrice:
    """Binary (digital) option prices."""

    cash_or_nothing_call: float
    cash_or_nothing_put: float
    asset_or_nothing_call: float
    asset_or_nothing_put: float


@dataclass(frozen=True, slots=True)
class PerpetualPutResult:
    """Perpetual American put pricing result."""

    price: float
    exercise_boundary: float
    lambda_2: float


_VEGA_FLOOR: float = 1e-12


# ── Validation ──────────────────────────────────────────────────────────────


def _validate_inputs(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,  # noqa: ARG001
    volatility: float,
    dividend_yield: float = 0.0,  # noqa: ARG001
) -> None:
    if spot <= 0:
        msg = "spot must be positive"
        raise ValueError(msg)
    if strike <= 0:
        msg = "strike must be positive"
        raise ValueError(msg)
    if time_to_expiry <= 0:
        msg = "time_to_expiry must be positive"
        raise ValueError(msg)
    if volatility <= 0:
        msg = "volatility must be positive"
        raise ValueError(msg)


# ── Internal helpers ────────────────────────────────────────────────────────


def _d1(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    return (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * math.sqrt(time_to_expiry))


def _d2(
    *,
    d1: float,
    volatility: float,
    time_to_expiry: float,
) -> float:
    return d1 - volatility * math.sqrt(time_to_expiry)


# ── Public API ──────────────────────────────────────────────────────────────


def price_european(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> OptionPrice:
    """Compute European call and put prices using the Black-Scholes formula.

    Supports continuous dividend yield (Merton 1973).

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        volatility: Annualized volatility of the underlying.
        dividend_yield: Continuous dividend yield (default 0).

    Returns:
        An OptionPrice with call and put values.

    Raises:
        ValueError: If any input is out of valid range.
    """
    _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    d1 = _d1(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    d2 = _d2(d1=d1, volatility=volatility, time_to_expiry=time_to_expiry)

    discount = math.exp(-risk_free_rate * time_to_expiry)
    div_discount = math.exp(-dividend_yield * time_to_expiry)

    call = spot * div_discount * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    put = strike * discount * norm.cdf(-d2) - spot * div_discount * norm.cdf(-d1)

    return OptionPrice(call=float(call), put=float(put))


def calculate_greeks(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Compute option Greeks (Delta, Gamma, Theta, Vega, Rho).

    Supports continuous dividend yield (Merton 1973).
    Theta is expressed per calendar day.  Vega and Rho are expressed per
    1 percentage-point move (i.e. divided by 100).

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        volatility: Annualized volatility of the underlying.
        is_call: True for a call option, False for a put.
        dividend_yield: Continuous dividend yield (default 0).

    Returns:
        A Greeks dataclass.

    Raises:
        ValueError: If any input is out of valid range.
    """
    _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    d1 = _d1(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    d2 = _d2(d1=d1, volatility=volatility, time_to_expiry=time_to_expiry)

    sqrt_t = math.sqrt(time_to_expiry)
    discount = math.exp(-risk_free_rate * time_to_expiry)
    div_discount = math.exp(-dividend_yield * time_to_expiry)
    pdf_d1 = norm.pdf(d1)

    # Gamma is the same for call and put
    gamma = float(div_discount * pdf_d1 / (spot * volatility * sqrt_t))

    # Vega (per 1 % vol move) is the same for call and put
    vega = float(spot * div_discount * pdf_d1 * sqrt_t / 100)

    if is_call:
        delta = float(div_discount * norm.cdf(d1))
        theta_annual = float(
            -(spot * div_discount * pdf_d1 * volatility) / (2 * sqrt_t)
            - risk_free_rate * strike * discount * norm.cdf(d2)
            + dividend_yield * spot * div_discount * norm.cdf(d1)
        )
        rho = float(strike * time_to_expiry * discount * norm.cdf(d2) / 100)
    else:
        delta = float(div_discount * (norm.cdf(d1) - 1))
        theta_annual = float(
            -(spot * div_discount * pdf_d1 * volatility) / (2 * sqrt_t)
            + risk_free_rate * strike * discount * norm.cdf(-d2)
            - dividend_yield * spot * div_discount * norm.cdf(-d1)
        )
        rho = float(-strike * time_to_expiry * discount * norm.cdf(-d2) / 100)

    theta = theta_annual / 365

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_volatility(
    *,
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    is_call: bool,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Solve for implied volatility using Newton-Raphson with bisection fallback.

    Supports continuous dividend yield (Merton 1973).

    Args:
        market_price: Observed market price of the option.
        spot: Current price of the underlying asset.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        is_call: True for a call option, False for a put.
        dividend_yield: Continuous dividend yield (default 0).
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        The implied volatility as a decimal (e.g. 0.20 for 20 %).

    Raises:
        ValueError: If inputs are invalid or the solver does not converge.
    """
    if market_price <= 0:
        msg = "market_price must be positive"
        raise ValueError(msg)
    if spot <= 0:
        msg = "spot must be positive"
        raise ValueError(msg)
    if strike <= 0:
        msg = "strike must be positive"
        raise ValueError(msg)
    if time_to_expiry <= 0:
        msg = "time_to_expiry must be positive"
        raise ValueError(msg)

    # Check arbitrage bounds (with dividend yield)
    discount = math.exp(-risk_free_rate * time_to_expiry)
    div_discount = math.exp(-dividend_yield * time_to_expiry)
    if is_call:
        intrinsic = max(spot * div_discount - strike * discount, 0.0)
        upper_bound = spot * div_discount
    else:
        intrinsic = max(strike * discount - spot * div_discount, 0.0)
        upper_bound = strike * discount

    if market_price < intrinsic - tol:
        msg = "market_price is below intrinsic value"
        raise ValueError(msg)
    if market_price > upper_bound + tol:
        msg = "market_price exceeds arbitrage upper bound"
        raise ValueError(msg)

    def _price_at_vol(vol: float) -> float:
        p = price_european(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=vol,
            dividend_yield=dividend_yield,
        )
        return p.call if is_call else p.put

    def _vega_at_vol(vol: float) -> float:
        d1 = _d1(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=vol,
            dividend_yield=dividend_yield,
        )
        div_disc = math.exp(-dividend_yield * time_to_expiry)
        return spot * div_disc * norm.pdf(d1) * math.sqrt(time_to_expiry)

    # Newton-Raphson
    vol = 0.2  # initial guess
    for _ in range(max_iter):
        price = _price_at_vol(vol)
        diff = price - market_price
        if abs(diff) < tol:
            return vol
        v = _vega_at_vol(vol)
        if v < _VEGA_FLOOR:
            break
        vol -= diff / v
        if vol <= 0:
            break

    # Bisection fallback
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price = _price_at_vol(mid)
        diff = price - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid

    msg = "implied volatility solver did not converge"
    raise ValueError(msg)


def price_surface(
    *,
    spots: np.ndarray,
    volatilities: np.ndarray,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    is_call: bool,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Compute a 2-D grid of option prices over spot x volatility.

    Args:
        spots: 1-D array of spot prices.
        volatilities: 1-D array of volatility values.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        is_call: True for call, False for put.
        dividend_yield: Continuous dividend yield (default 0).

    Returns:
        A 2-D numpy array of shape (len(volatilities), len(spots)).

    Raises:
        ValueError: If any input is out of valid range.
    """
    if strike <= 0:
        msg = "strike must be positive"
        raise ValueError(msg)
    if time_to_expiry <= 0:
        msg = "time_to_expiry must be positive"
        raise ValueError(msg)

    grid = np.empty((len(volatilities), len(spots)))
    for i, vol in enumerate(volatilities):
        for j, s in enumerate(spots):
            p = price_european(
                spot=float(s),
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=float(vol),
                dividend_yield=dividend_yield,
            )
            grid[i, j] = p.call if is_call else p.put
    return grid


# ── Extensions ──────────────────────────────────────────────────────────────


def price_discrete_dividend(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_proportion: float,
    num_dividends: int,
) -> OptionPrice:
    """Price European options with discrete proportional dividends.

    The spot is adjusted by ``S_adj = S * (1 - d)^n`` before applying the
    standard Black-Scholes formula.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        volatility: Annualized volatility of the underlying.
        dividend_proportion: Proportional dividend per payment (0 < d < 1).
        num_dividends: Number of dividend payments before expiry.

    Returns:
        An OptionPrice with call and put values.

    Raises:
        ValueError: If any input is out of valid range.
    """
    if dividend_proportion < 0 or dividend_proportion >= 1:
        msg = "dividend_proportion must be in [0, 1)"
        raise ValueError(msg)
    if num_dividends < 0:
        msg = "num_dividends must be non-negative"
        raise ValueError(msg)

    _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
    )

    adjusted_spot = spot * (1 - dividend_proportion) ** num_dividends

    return price_european(
        spot=adjusted_spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
    )


def price_binary(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> BinaryPrice:
    """Price binary (digital) options.

    Computes cash-or-nothing and asset-or-nothing prices for both calls
    and puts.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        volatility: Annualized volatility of the underlying.
        dividend_yield: Continuous dividend yield (default 0).

    Returns:
        A BinaryPrice with all four option type prices.

    Raises:
        ValueError: If any input is out of valid range.
    """
    _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    d1 = _d1(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    d2 = _d2(d1=d1, volatility=volatility, time_to_expiry=time_to_expiry)

    discount = math.exp(-risk_free_rate * time_to_expiry)
    div_discount = math.exp(-dividend_yield * time_to_expiry)

    return BinaryPrice(
        cash_or_nothing_call=float(discount * norm.cdf(d2)),
        cash_or_nothing_put=float(discount * norm.cdf(-d2)),
        asset_or_nothing_call=float(spot * div_discount * norm.cdf(d1)),
        asset_or_nothing_put=float(spot * div_discount * norm.cdf(-d1)),
    )


def price_fx_option(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
) -> OptionPrice:
    """Price FX options using the Garman-Kohlhagen (1983) model.

    This is equivalent to the Merton model with the foreign risk-free rate
    acting as the continuous dividend yield.

    Args:
        spot: Current exchange rate.
        strike: Strike exchange rate.
        time_to_expiry: Time to expiration in years.
        domestic_rate: Domestic risk-free interest rate.
        foreign_rate: Foreign risk-free interest rate.
        volatility: Annualized volatility of the exchange rate.

    Returns:
        An OptionPrice with call and put values.

    Raises:
        ValueError: If any input is out of valid range.
    """
    return price_european(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=domestic_rate,
        volatility=volatility,
        dividend_yield=foreign_rate,
    )


def calculate_fx_greeks(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    is_call: bool,
) -> Greeks:
    """Compute FX option Greeks using the Garman-Kohlhagen (1983) model.

    Args:
        spot: Current exchange rate.
        strike: Strike exchange rate.
        time_to_expiry: Time to expiration in years.
        domestic_rate: Domestic risk-free interest rate.
        foreign_rate: Foreign risk-free interest rate.
        volatility: Annualized volatility of the exchange rate.
        is_call: True for a call option, False for a put.

    Returns:
        A Greeks dataclass.

    Raises:
        ValueError: If any input is out of valid range.
    """
    return calculate_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=domestic_rate,
        volatility=volatility,
        is_call=is_call,
        dividend_yield=foreign_rate,
    )


def price_perpetual_put(
    *,
    spot: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> PerpetualPutResult:
    """Price a perpetual American put option (T -> infinity).

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        risk_free_rate: Annualized risk-free interest rate (must be > 0).
        volatility: Annualized volatility of the underlying.
        dividend_yield: Continuous dividend yield (default 0).

    Returns:
        A PerpetualPutResult with price, exercise boundary, and lambda_2.

    Raises:
        ValueError: If any input is out of valid range or r = 0.
    """
    if spot <= 0:
        msg = "spot must be positive"
        raise ValueError(msg)
    if strike <= 0:
        msg = "strike must be positive"
        raise ValueError(msg)
    if risk_free_rate <= 0:
        msg = "risk_free_rate must be strictly positive for perpetual put"
        raise ValueError(msg)
    if volatility <= 0:
        msg = "volatility must be positive"
        raise ValueError(msg)
    if dividend_yield < 0:
        msg = "dividend_yield must be non-negative"
        raise ValueError(msg)

    sigma2 = volatility**2
    drift_term = risk_free_rate - dividend_yield - sigma2 / 2
    lambda_2 = (
        -drift_term - math.sqrt(drift_term**2 + 2 * sigma2 * risk_free_rate)
    ) / sigma2

    exercise_boundary = lambda_2 * strike / (lambda_2 - 1)

    if spot <= exercise_boundary:
        price = strike - spot
    else:
        price = (strike / (1 - lambda_2)) * (
            (lambda_2 - 1) / lambda_2 * spot / strike
        ) ** lambda_2

    return PerpetualPutResult(
        price=float(price),
        exercise_boundary=float(exercise_boundary),
        lambda_2=float(lambda_2),
    )
