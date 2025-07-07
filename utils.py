# utils.py

import numpy as np
import torch
import tsl
from tsl.utils import ensure_list


def _masked_reduce(x, reduction, mask= None, nan_to_zero= False):
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    if reduction == 'none':
        if mask is not None:
            masked_idxs = torch.logical_not(mask)
            x[masked_idxs] = 0 if nan_to_zero else torch.nan
        return x
    if mask is not None:
        x = x[mask]
    if reduction == 'mean':
        return torch.mean(x)
    elif reduction == 'sum':
        return torch.sum(x)
    else:
        raise ValueError(
            f'reduction {reduction} not allowed, must be one of '
            "['mean', 'sum', 'none']."
        )


def mean_abs_error(y_hat, y, mask=None, reduction="mean", nan_to_zero=False):
    err = torch.abs(y_hat - y)
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def r_mean_abs_error(y_hat, y, mask = None, reduction = 'mean'):
    err = torch.square(y_hat - y)
    return torch.sqrt(_masked_reduce(err, reduction, mask))


def mean_abs_percent_error(y_hat, y, mask=None, reduction="mean", nan_to_zero=False):
    err = torch.abs((y_hat - y) / (y + tsl.epsilon))
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def MinMaxScaler(data):
    min_val = np.nanmin(data, axis=0)
    data = data - min_val
    max_val = np.nanmax(data, axis=0)
    max_val[max_val == 0] = 1e-8
    normalized_data = data / max_val
    norm_parameters = {"min_val": min_val, "max_val": max_val}
    return normalized_data, norm_parameters


def denormalize(data, norm_params):
    return data * norm_params['max_val'] + norm_params['min_val']


def calculate_metrics(y_true, y_pred, eval_mask):
    y_true_tensor = torch.from_numpy(np.asarray(y_true))
    y_pred_tensor = torch.from_numpy(np.asarray(y_pred))
    eval_mask_tensor = torch.from_numpy(np.asarray(eval_mask, dtype=bool))
    y_true_eval = y_true_tensor[eval_mask_tensor]
    y_pred_eval = y_pred_tensor[eval_mask_tensor]
    if len(y_true_eval) == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
    mae_metric = mean_abs_error(y_pred_eval, y_true_eval, reduction='mean', nan_to_zero=False)
    rmse_metric = r_mean_abs_error(y_pred_eval, y_true_eval, reduction='mean')
    mape_metric = mean_abs_percent_error(y_pred_eval, y_true_eval, reduction='mean', nan_to_zero=False)
    mae = mae_metric.item()
    rmse = rmse_metric.item()
    mape = mape_metric.item()
    return {'mae': mae, 'rmse': rmse, 'mape': mape}
