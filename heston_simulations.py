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
        
def cir_chi_sq_sample(heston_params: HestonParameters,
                      dt: float,
                      v_i: np.array,
                      n_simulations: int):
    """Samples chi_sqqred statistics for v_{i+1} conditional on 
       v_i and parameters of Hestom model. 
        
    Args:
        heston_params (HestonParameters): parameters of Heston model
        dt (float): time step 
        v_i: current volatility value
        n_simulations (int): number of simulations.
        
    Returns:
        np.array: sampled chi_squared statistics 
    """
    kappa, vbar, gamma = heston_params.kappa, heston_params.vbar, heston_params.gamma
    
    # YOUR COED HERE:
    # calculate \delta, c and \bar \kappa from Heston parameters and sample the value from
    # noncentral chi square distribution.
    # Hint 1: 
    #     Use np.random.noncentral_chisquare(...)
    #--------------------------------------------

    delta = 4.0 * kappa*vbar / (gamma **2)
    c = 1.0 / (4.0 * kappa) * gamma**2 * (1.0 - np.exp(-kappa * dt))
    kappaBar = 4.0 * kappa * v_i * np.exp(-kappa * dt) / ((gamma ** 2) * (1.0 - np.exp(- kappa * dt)))
    sample = c * np.random.noncentral_chisquare(delta, kappaBar, n_simulations)
    
    #--------------------------------------------
    
    return  sample



def simulate_paths_heston_euler(time: Union[float, np.ndarray],
                                n_simulations: int,
                                state: MarketState,
                                heston_params: HestonParameters) -> dict:
    """Simulates price and volatility process evaluating it at given time points 
       using Euler scheme. 
        
    Args:
        time (float or np.darray): time point(s) at which the price shoud be evaluated.
        n_simulations (int): number of simulations.
        state (MarketState): initial market state to start from.
        heston_params (HestonParameters): parameters of Heston model.
        
    Returns:
        dict: simulated asset and volatility paths.
    """
    
    # initialize market and model parameters
    r, s0 = state.interest_rate, state.stock_price
    
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, \
                                  heston_params.vbar, heston_params.gamma
    
    # initialize noise arrays
    Z1 = np.random.normal(size=(n_simulations, np.shape(time)[0]))
    Z2 = np.random.normal(size=(n_simulations, np.shape(time)[0]))
    
    #initialize zero array for volatility values
    V = np.zeros([n_simulations, np.shape(time)[0]])
    V[:, 0] = v0
    
    #initialize zero array for logprice values
    logS = np.zeros([n_simulations, np.shape(time)[0]])
    logS[:, 0] = np.log(s0) 
    
    # YOUR CODE HERE
    # Hint 1:
    #     You need to iterate in time and calculate logS[:, ind + 1] and V[:,ind + 1] given 
    #     logS[:, ind] and V[:, ind]. Notice that the V[:, ind] values may be negative, so you 
    #     may reflect the the positively or truncate it (take (max(V[:, ind], 0)).
    #
    # Hint 2: 
    #     You may use cycles if you wish. However, it is possible 
    #     to code it in a vectorized form.
    #----------------------------------------------------------------------------------------
    for ind in range(np.shape(time)[0] - 1):
        dt = time[ind + 1] - time[ind]
        logS[:, ind + 1] = logS[:, ind] + (r -  V[:, ind] / 2) * dt + \
                           np.power(dt * V[:, ind], 0.5) * (rho * Z1[:, ind] + np.sqrt(1 - rho**2) * Z2[:, ind])
        
        V[:, ind + 1] = V[:, ind] + kappa * (vbar - V[:, ind]) * dt + \
                        gamma * np.sqrt(V[:, ind] * dt) * Z1[:, ind]
        
        mask = V[:,ind + 1] < 0
        V[mask,ind + 1] *= -1
        
        #V[:,ind + 1] = np.maximum(V[:, ind + 1], 0.0)
    #---------------------------------------------------------------------------------------- 
    
    #put the calculated values into the dictionary
    simulated_paths = {"asset": np.exp(logS), "volatility": V}
    
    return simulated_paths



def simulate_paths_heston_andersen(time: Union[float, np.ndarray],
                                   n_simulations: int,
                                   state: MarketState,
                                   heston_params: HestonParameters) -> dict:
    """Simulates price and volatility process evaluating it at given time points 
       using Broadie-Kaya scheme. 
        
    Args:
        time (float or np.darray): time point(s) at which the price shoud be evaluated.
        n_simulations (int): number of simulations.
        state (MarketState): initial market state to start from.
        heston_params (HestonParameters): parameters of Heston model.
        
    Returns:
        dict: simulated asset and volatility paths
    """
    
    # initialize market and model parameters
    r, s0 = state.interest_rate,  state.stock_price
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, heston_params.vbar, heston_params.gamma
    
    # initialize noise array
    Z1 = np.random.normal(size=(n_simulations, np.shape(time)[0]))
    
    # initialize zero array for volatility values
    V = np.zeros([n_simulations, np.shape(time)[0]])
    V[:, 0] = v0
    
    # initialize zero array for log price values
    logS = np.zeros([n_simulations, np.shape(time)[0]])
    logS[:, 0] = np.log(s0)
    
    # YOUR CODE HERE
    # Hint 1:
    #     You need to iterate in time and calculate logS[:, ind + 1] and V[:,ind + 1] given 
    #     logS[:, ind] and V[:, ind]. Notice that the V[:, ind] values may be negative, so you 
    #     may reflect the the positively or truncate it (take (max(V[:, ind], 0)).
    #
    # Hint 2: 
    #     You may use cycles if you wish. However, it is possible 
    #     to code it in a vectorized form.
    # Hint 3:
    #     In order to get chi^2 sample, just call the cir_chi_sq_sample(...) function.
    #----------------------------------------------------------------------------------------
    
    for ind in range(np.shape(time)[0] - 1):
        dt = time[ind + 1] - time[ind]
        
        V[:,ind + 1] = cir_chi_sq_sample(heston_params, dt, V[:, ind], n_simulations)
        
        V[:,ind + 1] = np.maximum(V[:, ind + 1], 0.0)
        
        logS[:, ind + 1] = logS[:, ind] + (r - V[:, ind] / 2) * dt + \
                           (rho / gamma) * (V[:, ind + 1] - V[:, ind] - kappa * (vbar - V[:, ind]) * dt) + \
                           np.power((1 - rho**2) * dt * V[:, ind], 0.5) * Z1[:, ind] 
    #----------------------------------------------------------------------------------------
    
    simulated_paths = {"asset": np.exp(logS), "volatility": V}
    
    return simulated_paths