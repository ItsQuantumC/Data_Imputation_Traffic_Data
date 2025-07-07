# forecasting_data_loader.py

import numpy as np
import pandas as pd
from utils import MinMaxScaler

def create_forecasting_sequences(data, input_len, output_len):
    xs, ys = [], []
    for i in range(len(data) - input_len - output_len + 1):
        x = data[i:(i + input_len)]
        y = data[(i + input_len):(i + input_len + output_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_and_prepare_forecasting_data(imputed_file_path, original_data_file_path, input_len, output_len):
    imputed_df = pd.read_csv(imputed_file_path, index_col=0)
    ref_df = pd.read_csv(original_data_file_path, index_col='timestamp', parse_dates=True)
    if len(imputed_df) == len(ref_df):
        imputed_df.index = ref_df.index
    else:
        print("Warning: Imputed and reference data have different lengths. Aligning indices.")
        imputed_df.index = ref_df.index[-len(imputed_df):]
        
    df = imputed_df

    train_slice = slice('2023-02-28 23:00:00', '2023-09-12 17:00:00')
    valid_slice = slice('2023-09-12 18:00:00', '2023-11-16 23:00:00')
    test_slice = slice('2023-11-17 00:00:00', '2024-01-21 07:00:00')

    train_df = df.loc[train_slice]
    valid_df = df.loc[valid_slice]
    test_df = df.loc[test_slice]

    scaler = MinMaxScaler(train_df.values.astype(float))
    train_norm, norm_params = scaler
    valid_norm = (valid_df.values.astype(float) - norm_params['min_val']) / norm_params['max_val']
    test_norm = (test_df.values.astype(float) - norm_params['min_val']) / norm_params['max_val']

    train_x, train_y = create_forecasting_sequences(train_norm, input_len, output_len)
    valid_x, valid_y = create_forecasting_sequences(valid_norm, input_len, output_len)
    test_x, test_y = create_forecasting_sequences(test_norm, input_len, output_len)

    return {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "norm_params": norm_params
    }
