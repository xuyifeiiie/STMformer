import numpy

import torch
import numpy as np


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    # Calculate Mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # Masked MSE Loss
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    # Masked RMSE Loss
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_rmsle(preds, labels, null_val=np.nan):
    # Masked RMSLE Loss
    # loss = (torch.log(torch.abs(preds) + 1) - torch.log(torch.abs(labels) + 1)) ** 2
    preds = torch.log(torch.abs(preds) + 1)
    labels = torch.log(torch.abs(labels) + 1)
    return torch.sqrt(masked_mse(preds=preds,
                                 labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    # Calculate Mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # Masked MAE Loss
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, np.nan).item()
    mse = masked_mse(pred, real, np.nan).item()
    rmse = masked_rmse(pred, real, np.nan).item()
    rmsle = masked_rmsle(pred, real, np.nan).item()

    return mae, mse, rmse, rmsle


def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))

def RMSLE(pred, true):
    pred = torch.log(torch.abs(pred) + 1)
    true = torch.log(torch.abs(true) + 1)
    return torch.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return torch.mean(torch.abs((pred - true) / (true + 1e-10)))


def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / (true + 1e-10)))


def predict_metric(pred, true):
    mae = MAE(pred, true).cpu().detach().numpy()
    mse = MSE(pred, true).cpu().detach().numpy()
    rmse = RMSE(pred, true).cpu().detach().numpy()
    rmsle = RMSLE(pred, true).cpu().detach().numpy()
    mape = MAPE(pred, true).cpu().detach().numpy()
    mspe = MSPE(pred, true).cpu().detach().numpy()

    return mae, mse, rmse, rmsle, mape, mspe