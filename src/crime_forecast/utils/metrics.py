import numpy as np

def rmse_formula(y, yhat): y, yhat = np.asarray(y,float), np.asarray(yhat,float); return np.sqrt(np.mean((y-yhat)**2))

def mae_formula(y, yhat):  y, yhat = np.asarray(y,float), np.asarray(yhat,float); return np.mean(np.abs(y-yhat))

def mape(y, yhat):
    y, yhat = np.asarray(y,float), np.asarray(yhat,float); eps = 1e-8
    denom = np.where(y == 0, 1e-8, np.abs(y))
    return np.mean(np.abs((y - yhat)/denom)) * 100.0

def r2_formula(y, yhat):
    y, yhat = np.asarray(y,float), np.asarray(yhat,float)
    # R²
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - (ss_res/ss_tot if ss_tot != 0 else np.nan)
