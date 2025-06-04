import pandas_datareader.data as web
import yfinance as yf
import numpy as np
from torch.utils.data import Dataset
import torch

def get_asset_dataset(start, end):
    etfs = ['AGG', 'DBC', 'VTI']
    asset_names = etfs + ["VIX"]
    n_assets = len(asset_names)
    vix = web.DataReader('VIXCLS', 'fred',  start=start, end=end)
    close_df = yf.download(etfs, start=start, end=end)["Close"]
    close_df["VIX"] = vix

    returns_df =  close_df.pct_change()[1:]
    close_df = close_df[1:]
    return returns_df, close_df

def create_windows(data, indices, lookback, horizon):
    X, y = [], []
    index_list = []

    for i in range(len(indices) - lookback - horizon):
        # print(i)
        window_start = indices[i]
        input_end = indices[i + lookback]
        output_end = indices[i + lookback + horizon]

        x_window = data.loc[window_start:input_end].iloc[:-1].values  # shape: (lookback, features)
        y_window = data.loc[input_end:output_end].iloc[:horizon].values  # shape: (horizon, features)
        
        X.append(x_window)
        y.append(y_window)
        index_list.append(window_start)  # center timestamp for reference

    return np.array(X), np.array(y), index_list

def standardize(x, mean, std):
    return (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)

def destandardize(y, mean, std):

    return y * std.view(1, 1, -1) + mean.view(1, 1, -1)

class Jitter:
    def __init__(self, sigma):
        self.sigma = sigma#sigma.view(1, -1)  # shape (1, num_assets), for broadcasting

    def __call__(self, x):
        noise = torch.randn_like(x) * self.sigma  # apply per-asset noise
        return x + noise
    
class AugmentedTimeSeriesDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]
    

# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np

# def load_and_scale_data():
#     df = pd.read_csv("../data/DailyDelhiClimate.csv")
#     df['date'] = pd.to_datetime(df['date'])
#     df.set_index('date', inplace=True)
#     data = df["meantemp"].to_numpy()
#     scaler = StandardScaler()
#     data= scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
#     return data, scaler

# def create_dataset(data, window_size=10):
#     X, Y = [], []
#     for i in range(len(data) - window_size):
#         X.append(data[i:i+window_size])
#         Y.append(data[i+window_size])
#     return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

# def train_test_split(dataset, device):
#     X, Y = dataset
#     # Split into train/test
#     split = int(0.8 * len(X))
#     X_train, X_test = X[:split].to(device), X[split:].to(device)
#     Y_train, Y_test = Y[:split], Y[split:]

#     train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
#     test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32)
#     return train_loader, test_loader

# def load_ETFs():
#     ETFs = ["AGG", "DBC", "VIX", "VTI"]
#     assets = {}

#     for ETF in ETFs:
#         df = pd.read_csv("../data" + "/finance/" + ETF + "_History.csv")
#         # df['Close'] = df['Close/Last']
#         selected_cols = ["Date", "Open", "Close", "High", "Low"]
#         df = df[selected_cols]  
#         df['Date'] = pd.to_datetime(df['Date'])
#         df.set_index('Date', inplace=True)
#         df = df[(df.index >= '2006-04-02') & (df.index <= '2022-12-31')]
#         assets[ETF] = df
#     return assets

# def get_ETF_col(etfs, col):
#     result = pd.DataFrame()
#     for etf, df in etfs.items():
#         result[etf] = df[col]
#     return result

# def make_windows(data, lookback, horizon):
#     X_list, y_list = [], []


#     for i in range(lookback, len(data) - horizon + 1):
#         X_list.append(data[i - lookback: i, :])
#         y_list.append(data[i: i + horizon, :])

#     X = np.stack(X_list, axis = 0)
#     y = np.stack(y_list, axis = 0)

#     return X, y

# def train_test_split_indexes(N, percentage, lookback, horizon):
#     n_samples = N - lookback - horizon  + 1
#     split_ix = int(n_samples * percentage)
#     indices_train = list(range(split_ix))
#     indices_test = list(range(split_ix + lookback + horizon, n_samples))
#     return indices_train, indices_test