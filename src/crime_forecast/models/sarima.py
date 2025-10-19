from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(y_train, order, seasonal_order):
    model = SARIMAX(
        y_train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    return res

def sarima_forecast(res, steps):
    return res.get_forecast(steps=steps).predicted_mean