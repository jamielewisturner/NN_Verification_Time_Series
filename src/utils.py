import torch
import random
import numpy as np
import os
import pandas as pd

RESULT_PATH = "../results/"



def predict(model, scaler, X_test):
    return scaler.inverse_transform(model(X_test).cpu().detach().numpy())


def load_or_create_df(df_path, create_fn, force_recreate=False):

    df_path = RESULT_PATH + df_path

    if os.path.exists(df_path) and not force_recreate:
        print(f"🔄 Loading DataFrame from {df_path}")
        df = pd.read_csv(df_path, index_col=0)
    else:
        print(f"🧪 Creating new DataFrame")
        df = create_fn()
        print(f"💾 Saving DataFrame to {df_path}")
        df.to_csv(df_path)
    
    return df