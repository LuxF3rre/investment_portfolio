"""Black-Scholes-Merton European option pricing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = [
    "Greeks",
    "OptionPrice",
    "calculate_greeks",
    "implied_volatility",
    "price_european",
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


_VEGA_FLOOR: float = 1e-12


# ── Validation ──────────────────────────────────────────────────────────────


def _validate_inputs(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
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
    if risk_free_rate < 0:
        msg = "risk_free_rate must be non-negative"
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
) -> float:
    return (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
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
) -> OptionPrice:
    """Compute European call and put prices using the Black-Scholes formula.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        volatility: Annualized volatility of the underlying.

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
    )

    d1 = _d1(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
    )
    d2 = _d2(d1=d1, volatility=volatility, time_to_expiry=time_to_expiry)

    discount = math.exp(-risk_free_rate * time_to_expiry)

    call = spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    put = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return OptionPrice(call=float(call), put=float(put))


def calculate_greeks(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    is_call: bool,
) -> Greeks:
    """Compute option Greeks (Delta, Gamma, Theta, Vega, Rho).

    Theta is expressed per calendar day.  Vega and Rho are expressed per
    1 percentage-point move (i.e. divided by 100).

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        volatility: Annualized volatility of the underlying.
        is_call: True for a call option, False for a put.

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
    )

    d1 = _d1(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
    )
    d2 = _d2(d1=d1, volatility=volatility, time_to_expiry=time_to_expiry)

    sqrt_t = math.sqrt(time_to_expiry)
    discount = math.exp(-risk_free_rate * time_to_expiry)
    pdf_d1 = norm.pdf(d1)

    # Gamma is the same for call and put
    gamma = float(pdf_d1 / (spot * volatility * sqrt_t))

    # Vega (per 1 % vol move) is the same for call and put
    vega = float(spot * pdf_d1 * sqrt_t / 100)

    if is_call:
        delta = float(norm.cdf(d1))
        theta_annual = float(
            -(spot * pdf_d1 * volatility) / (2 * sqrt_t)
            - risk_free_rate * strike * discount * norm.cdf(d2)
        )
        rho = float(strike * time_to_expiry * discount * norm.cdf(d2) / 100)
    else:
        delta = float(norm.cdf(d1) - 1)
        theta_annual = float(
            -(spot * pdf_d1 * volatility) / (2 * sqrt_t)
            + risk_free_rate * strike * discount * norm.cdf(-d2)
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
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Solve for implied volatility using Newton-Raphson with bisection fallback.

    Args:
        market_price: Observed market price of the option.
        spot: Current price of the underlying asset.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        is_call: True for a call option, False for a put.
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
    if risk_free_rate < 0:
        msg = "risk_free_rate must be non-negative"
        raise ValueError(msg)

    # Check arbitrage bounds
    discount = math.exp(-risk_free_rate * time_to_expiry)
    if is_call:
        intrinsic = max(spot - strike * discount, 0.0)
        upper_bound = spot
    else:
        intrinsic = max(strike * discount - spot, 0.0)
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
        )
        return p.call if is_call else p.put

    def _vega_at_vol(vol: float) -> float:
        d1 = _d1(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=vol,
        )
        return spot * norm.pdf(d1) * math.sqrt(time_to_expiry)

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
) -> np.ndarray:
    """Compute a 2-D grid of option prices over spot x volatility.

    Args:
        spots: 1-D array of spot prices.
        volatilities: 1-D array of volatility values.
        strike: Option strike price.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annualized risk-free interest rate.
        is_call: True for call, False for put.

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
    if risk_free_rate < 0:
        msg = "risk_free_rate must be non-negative"
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
            )
            grid[i, j] = p.call if is_call else p.put
    return grid
