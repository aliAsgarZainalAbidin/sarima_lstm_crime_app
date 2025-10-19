import pytest
from src.crime_forecast.models.sarima import fit_sarima, sarima_forecast
from src.crime_forecast.models.lstm import ResidualLSTM, fit_lstm_and_predict
import pandas as pd
import numpy as np

# Sample data for testing
@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2020-01-01", periods=24, freq="M")
    values = np.random.poisson(lam=100, size=24)
    return pd.DataFrame({"date": dates, "value": values}).set_index("date")

def test_fit_sarima(sample_data):
    train_data = sample_data["value"][:-6]
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    
    model = fit_sarima(train_data, order, seasonal_order)
    assert model is not None
    assert hasattr(model, 'fittedvalues')

def test_sarima_forecast(sample_data):
    train_data = sample_data["value"][:-6]
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    
    model = fit_sarima(train_data, order, seasonal_order)
    forecast = sarima_forecast(model, steps=6)
    
    assert len(forecast) == 6
    assert forecast.index.equals(pd.date_range(start=train_data.index[-1] + pd.DateOffset(months=1), periods=6, freq='M'))

def test_fit_lstm_and_predict(sample_data):
    residual_series = sample_data["value"][:-6] - np.mean(sample_data["value"][:-6])
    horizon = 6
    window = 3
    hidden = 16
    lr = 0.001
    batch_size = 2
    epochs = 10
    patience = 2
    
    preds = fit_lstm_and_predict(residual_series, horizon, window, hidden, lr, batch_size, epochs, patience)
    
    assert len(preds) == horizon
    assert isinstance(preds, np.ndarray)

def test_residual_lstm_forward():
    model = ResidualLSTM(hidden=16)
    input_tensor = torch.randn(1, 6, 1)  # Batch size of 1, sequence length of 6, feature size of 1
    output = model(input_tensor)
    
    assert output.shape == (1, 1)  # Output should be of shape (batch_size, output_size)