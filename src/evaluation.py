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


def get_portfolio_value(model, data, rebalance_freq):
    portfolio_values = []
    x, y = data[0], data[1].numpy()
    model.eval()
    n_days, n_assets = y.shape
    total_value = 1
    weights = []
    asset_values = np.zeros(n_assets)

    for t in range(0, n_days, rebalance_freq):
        current_weights = model(x[t:t+1]).detach().cpu().numpy().flatten()
        asset_values = total_value * current_weights
        block_len = min(rebalance_freq, n_days - t)

        for j in range(block_len):
            d = t + j
            asset_values = asset_values * (1 + y[d])
            pv = asset_values.sum()
            weights.append(asset_values / pv)
            portfolio_values.append(asset_values.copy())

        total_value = portfolio_values[-1].sum()

    portfolio_values = np.array(portfolio_values)
    weights = np.vstack(weights)

    return portfolio_values.sum(axis=1), weights[:n_days], portfolio_values