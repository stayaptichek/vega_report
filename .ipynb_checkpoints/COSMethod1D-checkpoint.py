import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.optimize import root_scalar, brentq
from dataclasses import dataclass, field
from typing import Union, Callable, Optional
from copy import deepcopy
import warnings
from  scipy.stats import norm
import scipy.stats as st
import scipy.optimize as optimize
import enum
warnings.filterwarnings("ignore")

@dataclass
class StockOption:
    strike_price: Union[float, np.ndarray]
    expiration_time: Union[float, np.ndarray]  # in years
    is_call: bool

@dataclass
class CallStockOption(StockOption):
    def __init__(self, strike_price, expiration_time):
        super().__init__(strike_price, expiration_time, True)


@dataclass
class PutStockOption(StockOption):
    def __init__(self, strike_price, expiration_time):
        super().__init__(strike_price, expiration_time, False)

@dataclass
class MarketState:
    stock_price: Union[float, np.ndarray]
    interest_rate: Union[float, np.ndarray]  # r, assume constant
    dividends_rate: Union[float, np.ndarray] # q, assume constant

@dataclass
class HestonParameters:
    kappa:  Union[float, np.ndarray]
    gamma:  Union[float, np.ndarray]
    rho:  Union[float, np.      ndarray]
    vbar:  Union[float, np.ndarray]
    v0:  Union[float, np.ndarray]


def CallPutOptionPriceCOSMthd(сharacteristic_function,
                              is_call: bool,
                              state: MarketState,
                              strikes: Union[float, np.array],
                              time_to_maturity: float,
                              N: int, L: float):
    """
        Calculates option prices using COS method.

    Args:
        characteristic_function: characteristic function from Heston model.
        is_call: type of option (True if Call, else False)
        state: MarketState
        strikes: grid of option strikes
        L: scaling parameter for left and right boundaries of integration
        N: number of components in Fourier sum
    """

    S_0 = state.stock_price
    r = state.interest_rate

    if strikes is not np.array:
        strikes = np.array(strikes).reshape([len(strikes),1])

    i = np.complex(0.0,1.0)
    X_0 = np.log(S_0 / strikes)

    # Truncation domain
    a = 0.0 - L * np.sqrt(time_to_maturity)
    b = 0.0 + L * np.sqrt(time_to_maturity)

    # Summation from k = 0 to k=N-1
    k = np.linspace(0, N-1, N).reshape([N, 1])
    u = k * np.pi / (b - a);

    # Determine coefficients for put prices
    H_k = CallPutCoefficients(is_call, a, b, k)
    mat = np.exp(i * np.outer((X_0 - a) , u))
    temp = сharacteristic_function(u, time_to_maturity) * H_k
    temp[0] = 0.5 * temp[0]
    value = np.exp(-r * time_to_maturity) * strikes * np.real(mat.dot(temp))

    return value

# Determine coefficients for put prices

def CallPutCoefficients(is_call, a, b, k):
    """Calculates options coefficients in Fourier expansion.

    Args:
        is_call: type of option (True if Call, else False)
        a (float): left boundary of integration for Fourier coefficients
        b (float): left boundary of integration for Fourier coefficients
        k: linspace of indexes

    """
    if is_call:
        c, d = 0.0, b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k, Psi_k = coef["chi"], coef["psi"]

        if a < b and b < 0.0:
            H_k = np.zeros([len(k), 1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)
        return H_k

    c, d = a, 0.0
    coef = Chi_Psi(a, b, c, d, k)
    Chi_k, Psi_k = coef["chi"], coef["psi"]
    H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)

    return H_k

def Chi_Psi(a, b, c, d, k):
    """Calculates cosine series coefficients chi_k and psi_k

    Args:
        a (float): left boundary of integration for Fourier coefficients
        b (float): left boundary of integration for Fourier coefficients
        c (flaot): left boundary of integration for chi_k and psi_k
        d (flaot): right boundary of integration for chi_k and psi_k
        k: linspace of indexes

    Returns:
        dict: dict with chi and psi arrays of values
    """
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c

    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0))
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)

    value = {"chi":chi, "psi":psi}

    return value


def BS_Call_Put_Option_Price(is_call: bool,
                             state: MarketState,
                             strikes: Union[float, np.array],
                             sigma: float,
                             time_to_maturity: float):
    """
    Black-Scholes Call-Put option price formula

    Args:
        is_call: type of option (True if Call, else False)
        state: MarketState
        strikes: grid of option strikes
        sigma: asset volatility
        time_to_maturity (float or np.array): options time to expiration

    Returns:
        Union[float, np.array]: option price
    """
    S_0, r = state.stock_price, state.interest_rate

    if strikes is list:
        strikes = np.array(strikes).reshape([len(strikes),1])

    d1    = (np.log(S_0 / strikes) + (r + 0.5 * np.power(sigma, 2.0)) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    d2    = d1 - sigma * np.sqrt(time_to_maturity)

    if is_call:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * strikes * np.exp(-r * time_to_maturity)
    else:
        value = st.norm.cdf(-d2) * strikes * np.exp(-r * time_to_maturity) - st.norm.cdf(-d1) * S_0

    return value

# Implied volatility method

def ImpliedVolatility(is_call: bool,
                      option_market_price: float,
                      strikes: Union[float, np.array],
                      time_to_maturity: float,
                      state: MarketState):
    """
    Black-Scholes implied volatility calculation.

    Args:
        is_call: type of option (True if Call, else False)
        option_market_price: market price of option
        strikes: grid of option strikes
        time_to_maturity (float or np.array): options time to expiration
        state: MarketState


    Returns:
        float: implied volatility
    """

    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigma_grid = np.linspace(0, 2, 200)
    option_prices_grid = BS_Call_Put_Option_Price(is_call=is_call, state=state,
                                                  strikes=strikes, sigma=sigma_grid,
                                                  time_to_maturity=time_to_maturity)

    initial_sigma = np.interp(option_market_price, option_prices_grid, sigma_grid)

    #print("Initial volatility = {0}".format(initial_sigma))

    # Use already determined input for the local-search (final tuning)
    function = lambda sigma: np.power(BS_Call_Put_Option_Price(is_call, state, strikes, sigma, time_to_maturity) - option_market_price, 1.0)
    implied_volatility = optimize.newton(function, initial_sigma, tol=1e-12)

    #print("Final volatility = {0}".format(implied_volatility))

    return implied_volatility

def ChFHestonModel(state: MarketState,
                   heston_parameters: HestonParameters,
                   time_to_maturity: float):
    """Calculates characteristic function for Heston model.

    Args:
        state (MarketState): initial market state to start from.
        heston_params (HestonParameters): parameters of Heston model.
        time_to_maturity (float or np.array): options time to expiration

    Returns:
        Callable[[float, float], complex]: characteritstic function
    """
    r = state.interest_rate
    kappa, gamma, vbar, v0, rho = heston_parameters.kappa, heston_parameters.gamma, heston_parameters.vbar, \
                                  heston_parameters.v0, heston_parameters.rho


    i = np.complex(0.0, 1.0)

    D1 = lambda u: np.sqrt(np.power(kappa - gamma * rho * i * u, 2)+(u**2 + i*u) * gamma**2)

    g  = lambda u: (kappa - gamma * rho * i * u - D1(u)) / (kappa - gamma * rho * i * u + D1(u))

    C  = lambda u, tau: (1.0 - np.exp(-D1(u) * time_to_maturity))/(gamma**2 *(1.0 - g(u) * np.exp(-D1(u) * time_to_maturity)))\
                        *(kappa - gamma * rho * i * u - D1(u))

    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A  = lambda u, tau: r * i* u * time_to_maturity + kappa * vbar * time_to_maturity/(gamma**2) * (kappa - gamma * rho * i * u - D1(u))\
                        - 2 * kappa * vbar / (gamma**2) * np.log((1.0 - g(u) * np.exp(-D1(u) * time_to_maturity))/(1.0  - g(u)))

    # Characteristic function for the Heston model
    characteristic_function = lambda u, tau: np.exp(A(u, time_to_maturity) + C(u, time_to_maturity) * v0)
    return characteristic_function

def OptionPriceWithCosMethodHelp(is_call: bool,
                                 time_to_maturity: float,
                                 strikes: Union[float,  np.array],
                                 state: MarketState,
                                 heston_parameters: HestonParameters,
                                 N: int, L: float):
    """Wrapper for COSMethod function.

    Args:
        is_call: type of option (True if Call, else False)
        time_to_maturity (float or np.array): options time to expiration
        strikes: grid of option strikes
        state: MarketState
        heston_parameters: HestonParameters
        L: scaling parameter for left and right boundares of integration
        N: number of components in Fourie sum

    Returns:
        float: option price.
    """

    # Characteristic function for the Heston model
    characteristic_function = ChFHestonModel(state, heston_parameters, time_to_maturity)

    OptPrice = CallPutOptionPriceCOSMthd(characteristic_function, is_call, state,
                                         strikes, time_to_maturity, N, L)
    return OptPrice

def EUOptionPriceFromMCPaths(is_call: bool,
                             stock_price: Union[float, np.array],
                             strikes: Union[float, np.array],
                             time_to_maturity: float,
                             interest_rate: float):
    """Monte Carlo estimator for option price.

    Args:
        is_call: type of option (True if Call, else False)
        stock_price: terminal asset price at maturity
        strikes: grid of option strikes
        time_to_maturity (float): options time to expiration
        intererst_rate (float): market interest rate

    Returns:
        float: option price.
    """

    if is_call:
        return np.exp(-interest_rate * time_to_maturity) * np.mean(np.maximum(stock_price - strikes, 0.0))

    return np.exp(-interest_rate * time_to_maturity) * np.mean(np.maximum(strikes - stock_price, 0.0))

