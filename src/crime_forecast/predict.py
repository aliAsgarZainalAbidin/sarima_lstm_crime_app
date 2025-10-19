import numpy as np
import pandas as pd
import torch
from pathlib import Path
from crime_forecast.models.persistence import load_sarima, load_lstm_state, load_scaler, load_metadata
from datetime import datetime
import math

def months_between(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

def forecast_with_hybrid(sarima_path, lstm_model_class, lstm_state_path, scaler_path, hist_series, start_month, end_month, device="cpu", window=12):
    """
    sarima_path: path to saved SARIMA pickle
    lstm_model_class: callable that returns an uninitialized LSTM model of same architecture used in training
    lstm_state_path: .pth path
    scaler_path: joblib scaler path used to scale residuals (can be None)
    hist_series: pandas Series indexed by datetime (monthly) with historical values
    start_month / end_month: date or datetime.date objects (month/year used)
    device: torch device string
    window: LSTM input window length used in training

    Returns: DataFrame with monthly dates and predicted values (sum of SARIMA + LSTM residual)
    """
    sarima = load_sarima(sarima_path)
    lstm = load_lstm_state(lstm_model_class, lstm_state_path, device=device)
    scaler = load_scaler(scaler_path) if scaler_path else None

    # ensure monthly index
    s = hist_series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("MS").fillna(method="ffill")

    # horizon in months (inclusive)
    if isinstance(start_month, (datetime, pd.Timestamp)):
        start = pd.Timestamp(start_month).to_period("M").to_timestamp()
    else:
        start = pd.Timestamp(start_month).to_period("M").to_timestamp()
    if isinstance(end_month, (datetime, pd.Timestamp)):
        end = pd.Timestamp(end_month).to_period("M").to_timestamp()
    else:
        end = pd.Timestamp(end_month).to_period("M").to_timestamp()
    horizon = months_between(start.date(), end.date())

    # 1) SARIMA point forecast
    try:
        sarima_pred = sarima.get_forecast(steps=horizon)
        sarima_mean = pd.Series(sarima_pred.predicted_mean.values, index=pd.date_range(s.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"))
    except Exception:
        # fallback: use sarima.predict(start, end) if available
        last = s.index.max()
        future_idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        sarima_mean = pd.Series([float(s.mean())]*horizon, index=future_idx)

    # 2) prepare residual series (historical residual = actual - sarima_fitted)
    residuals = None
    try:
        if hasattr(sarima, "fittedvalues"):
            fitted = pd.Series(sarima.fittedvalues, index=s.index[: len(sarima.fittedvalues)])
            # align lengths
            fitted = fitted.reindex(s.index).fillna(method="ffill")
            residuals = s - fitted
        else:
            residuals = pd.Series(np.zeros(len(s)), index=s.index)
    except Exception:
        residuals = pd.Series(np.zeros(len(s)), index=s.index)

    # 3) use LSTM to predict residuals iteratively
    preds_res = []
    res_arr = residuals.values.astype(float)
    # normalize if scaler provided
    if scaler is not None:
        res_scaled = scaler.transform(res_arr.reshape(-1,1)).flatten()
    else:
        res_scaled = (res_arr - np.mean(res_arr)) / (np.std(res_arr) + 1e-8)

    # take last window values
    buf = list(res_scaled[-window:]) if len(res_scaled) >= window else list(np.pad(res_scaled, (window - len(res_scaled), 0), "constant", constant_values=0.0))

    lstm.to(device)
    lstm.eval()
    with torch.no_grad():
        for step in range(horizon):
            x = torch.tensor(buf[-window:], dtype=torch.float32).reshape(1, window, 1).to(device)
            out = lstm(x)  # expect model returns single-step scalar tensor
            if isinstance(out, tuple):
                out = out[0]
            pred_val = out.detach().cpu().numpy().reshape(-1)[0]
            preds_res.append(float(pred_val))
            buf.append(pred_val)

    # inverse-transform residual predictions
    if scaler is not None:
        preds_res_arr = scaler.inverse_transform(np.array(preds_res).reshape(-1,1)).flatten()
    else:
        # inverse of z-score used above
        mu = np.mean(res_arr)
        sigma = np.std(res_arr) + 1e-8
        preds_res_arr = preds_res * sigma + mu

    # 4) combine sarima + residual predictions and return only requested months
    combined = sarima_mean.copy()
    combined.iloc[:] = combined.values + preds_res_arr[:len(combined)]

    # trim to requested start..end range (in case pipeline/horizon mismatched)
    mask = (combined.index >= start) & (combined.index <= end)
    result = pd.DataFrame({"date": combined.index, "predicted": combined.values})
    result = result.loc[mask].reset_index(drop=True)
    return result