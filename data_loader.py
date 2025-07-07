# data_loader.py

import numpy as np
import pandas as pd
from utils import MinMaxScaler

def create_sequences(data, seq_len):
    xs = []
    if data.shape[0] < seq_len:
        return np.array([])
    for i in range(data.shape[0] - seq_len + 1):
        xs.append(data[i:(i + seq_len)])
    return np.array(xs)


def introduce_artificial_nans(data, natural_mask, rate):
    artificial_mask = np.zeros_like(data, dtype=bool)
    observed_indices = np.where(~natural_mask)
    num_to_mask = int(len(observed_indices[0]) * rate)
    if num_to_mask == 0:
        return data, artificial_mask
    mask_indices = np.random.choice(len(observed_indices[0]), num_to_mask, replace=False)
    rows, cols = observed_indices[0][mask_indices], observed_indices[1][mask_indices]
    artificial_mask[rows, cols] = True
    data_with_artificial_nans = data.copy()
    data_with_artificial_nans[artificial_mask] = np.nan
    return data_with_artificial_nans, artificial_mask


def create_time_gap_matrix(mask):
    rows, cols = mask.shape
    t = np.ones_like(mask, dtype=float)
    for j in range(cols):
        for k in range(1, rows):
            if mask[k, j] == 0:
                t[k, j] = t[k - 1, j] + 1
    return t


def load_and_prepare_data(file_path, seq_len, artificial_missing_rate, grin_path, mean_path):
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    train_slice = slice('2023-02-28 23:00:00', '2023-09-12 17:00:00')
    valid_slice = slice('2023-09-12 18:00:00', '2023-11-16 23:00:00')
    test_slice = slice('2023-11-17 00:00:00', '2024-01-21 07:00:00')
    train_df = df.loc[train_slice]
    valid_df = df.loc[valid_slice]
    test_df = df.loc[test_slice]
    test_indices = df.index.get_indexer_for(test_df.index)
    natural_nan_mask_train = train_df.isna().values
    natural_nan_mask_valid = valid_df.isna().values
    natural_nan_mask_test = test_df.isna().values
    _, norm_params = MinMaxScaler(train_df.values.astype(float))

    def apply_normalization(df_vals, params):
        return (df_vals - params['min_val']) / params['max_val']

    train_norm = apply_normalization(train_df.values, norm_params)
    train_norm_means = np.nanmean(train_norm, axis=0)
    train_norm - np.nan_to_num(train_norm)
    valid_norm = apply_normalization(valid_df.values, norm_params)
    test_norm = apply_normalization(test_df.values, norm_params)
    original_test_data = test_norm.copy()
    train_norm_w_artificial, artificial_nan_mask_train = introduce_artificial_nans(train_norm, natural_nan_mask_train, artificial_missing_rate)
    test_norm_w_artificial, artificial_nan_mask_test = introduce_artificial_nans(test_norm, natural_nan_mask_test, artificial_missing_rate)
    train_x_full = np.nan_to_num(train_norm_w_artificial)
    valid_x_full = np.nan_to_num(valid_norm)
    test_x_full = np.nan_to_num(test_norm_w_artificial)
    train_m_full = 1 - (natural_nan_mask_train | artificial_nan_mask_train).astype(float)
    valid_m_full = 1 - natural_nan_mask_valid.astype(float)
    test_m_full = 1 - (natural_nan_mask_test | artificial_nan_mask_test).astype(float)
    train_t_full = create_time_gap_matrix(train_m_full)
    valid_t_full = create_time_gap_matrix(valid_m_full)
    test_t_full = create_time_gap_matrix(test_m_full)
    data_dict = {
        "train": tuple(map(create_sequences, [train_x_full, train_m_full, train_t_full, train_norm], [seq_len]*4)),
        "valid": tuple(map(create_sequences, [valid_x_full, valid_m_full, valid_t_full, np.nan_to_num(valid_norm)], [seq_len]*4)),
        "test": tuple(map(create_sequences, [test_x_full, test_m_full, test_t_full, original_test_data], [seq_len]*4)),
        "test_masks": {"artificial": create_sequences(artificial_nan_mask_test.astype(float), seq_len)},
        "norm_params": norm_params,
        "train_norm_means": train_norm_means
    }
    grin_full_df = pd.read_csv(grin_path, index_col=0)
    mean_full_df = pd.read_csv(mean_path, index_col='timestamp', parse_dates=True)
    grin_test_vals = grin_full_df.iloc[test_indices].values
    mean_test_vals = mean_full_df.loc[test_df.index].values
    data_dict["baselines"] = {
        "grin": create_sequences(apply_normalization(grin_test_vals, norm_params), seq_len),
        "mean": create_sequences(apply_normalization(mean_test_vals, norm_params), seq_len)
    }
    return data_dict
