import numpy as np
import torch

def sharpe_ratio(returns):
    mean_return = np.mean(returns)
    volatility = np.std(returns)
    return mean_return / volatility

def max_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / (peak + 1e-8)
    return -drawdowns.min()

def entropy(weights):
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)  # (T,)
    return entropy

def mean_entropy(weights):
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)  # (T,)
    return entropy.mean().item()