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


def characteristic_function_Heston_model(state: MarketState, 
                                         heston_parameters: HestonParameters) -> Callable[[float], complex]:
    """Calculates characteristic function for Heston model. 
        
    Args:
        state (MarketState): initial market state to start from.
        heston_params (HestonParameters): parameters of Heston model.
        expiration_time (float or np.array): options time to expiration
    
    Returns:
        Callable[[float, float], complex]: characteritstic function
        
    """
    r = state.interest_rate
    kappa, gamma, vbar, v0, rho = heston_parameters.kappa, heston_parameters.gamma, heston_parameters.vbar,\
                                  heston_parameters.v0, heston_parameters.rho
        
    i = complex(0.0, 1.0)
    
    D1 = lambda u: np.sqrt(np.power(kappa - gamma * rho * i * u, 2) + (u**2 + i * u) * gamma**2)
    
    g  = lambda u: (kappa - gamma * rho * i * u - D1(u)) / (kappa - gamma * rho * i * u + D1(u))
    
    C  = lambda u, T: (1.0 - np.exp(- D1(u) * T))/(gamma * gamma * (1.0 - g(u) * np.exp(- D1(u) * T)))\
                    * (kappa - gamma * rho * i * u - D1(u))

    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method

    A  = lambda u, T: r * i * u * T + \
                        kappa * vbar * T / (gamma**2)  * (kappa - gamma * rho * i * u - D1(u)) \
                        - 2. * kappa * vbar /(gamma**2) * np.log((1.0 - g(u) * np.exp(- D1(u) * T)) / (1.0 - g(u)))

    # Characteristic function for the Heston model    

    cf = lambda u, T: np.exp(A(u, T) + C(u, T) * v0)
    
    return cf


def cosine_series_coefficients(a: float, b: float, 
                               c: float, d: float, 
                               k: np.array) -> dict:
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
    
    value = {"chi": chi, "psi": psi}
    
    return value



def options_coefficients(option_type: bool,
                         a: float, b: float,
                         k: int):
    """Calculates options coefficients in Fourier expansion. 
    
    Args:
        option_type: type of option (True if Call, else False)
        a (float): left boundary of integration for Fourier coefficients
        b (float): left boundary of integration for Fourier coefficients
        k: linspace of indexes
        
    """
    if option_type:  
        c, d = 0.0, b
        coef = cosine_series_coefficients(a, b, c, d, k)
        Chi_k, Psi_k = coef["chi"], coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([k.shape[0], 1])
            
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)  
    else:
        c, d = a, 0.0
        coef = cosine_series_coefficients(a, b, c, d, k)
        Chi_k, Psi_k = coef["chi"], coef["psi"]
        
        H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    
    return H_k  

def COS_method(char_function: Callable[[complex], complex],
               option: StockOption,
               state: MarketState,
               L: float,
               N: int,
               greek_flag: bool=False):
    """
        Calculates option prices using COS method.
            
    Args: 
        char_function: characteristic function from Heston model.
        option: option (tyoe of option, maturities, strikes)
        state: MarketState
        L: scaling parameter for left and right boundares of integration
        N: number of components in Fourie sum
        greek_flag: whether to calculate delta or not
    """
                   
    S_0, r = state.stock_price, state.interest_rate
    K, is_call, T = option.strike_price, option.is_call, option.expiration_time
    
    #print(K.shape, T.shape)
    
    i = complex(0., 1.)
    
    if isinstance(K, int):
        X_0 = np.log(S_0 / K)
    else:
        X_0 = np.log(S_0 / K).reshape(K.shape[1], 1)
    
    if not isinstance(K, int):
        assert X_0.shape == (K.shape[1], 1)
    
    ans = []
    greeks = []
        
    for t in T:
        a, b = -L * np.sqrt(t), L * np.sqrt(t)
        k_space = np.arange(N).reshape(N, 1)        
        u_space = k_space * np.pi * 1./ (b - a) #k pi / (b-a)
        
        exp_space = np.nan_to_num(np.exp(i * np.outer((X_0 - a), u_space)), nan=0.)
        exp_space_greeks = np.exp(i * np.outer((X_0 - a), u_space))
        exp_space_greeks = exp_space_greeks * u_space.T * i
                                
        if not isinstance(K, int):
            assert np.shape(exp_space) == (K.shape[1], N)

        U_k = options_coefficients(is_call, a, b, k_space)
        
        #calculate options prices
        chi_values = char_function(u_space, t) * U_k
        chi_values[0] = 0.5 * chi_values[0] 
        chi_values = np.nan_to_num(chi_values, nan=0.0)
        sum_ = np.real(exp_space.dot(chi_values))
        ans.append(np.exp(-r * t) * (K.reshape(K.shape[1], 1) * sum_))
        
        #calcaulate greeks (deltas)
        greeks_ = np.nan_to_num(np.real(exp_space_greeks.dot(chi_values)), 0.)
        greeks.append(np.exp(-r * t) * (K.reshape(K.shape[1], 1) * greeks_))
    
    if greek_flag:
        return np.nan_to_num(np.array(ans), nan=0.0), np.array(greeks) / S_0
    
    return np.nan_to_num(np.array(ans), nan=0.0)

