import pytest
from src.crime_forecast.pipeline import run_pipeline
import pandas as pd


@pytest.fixture
def sample_data():
    data = {
        "date": pd.date_range(start="2020-01-01", periods=12, freq="M"),
        "value": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "waktu_kejadian": pd.date_range(start="2020-01-10", periods=12, freq="M"),
        "tkp": ["Location A"] * 6 + ["Location B"] * 6,
        "jenis_kejahatan": ["Type 1"] * 6 + ["Type 2"] * 6,
        "jumlah_kejadian": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
    return pd.DataFrame(data)


def test_run_pipeline(sample_data):
    result = run_pipeline(
        df_raw=sample_data,
        date_col_name="date",
        value_col_name="value",
        do_outlier_iqr=True,
        waktu_col_name="waktu_kejadian",
        tkp_col_name="tkp",
        jenis_col_name="jenis_kejahatan",
        jumlah_col_name="jumlah_kejadian",
        coords_file=None,
        geojson_file=None,
        test_len=3,
        use_auto_arima=False,
        order_p=1,
        order_d=1,
        order_q=1,
        seas_P=1,
        seas_D=1,
        seas_Q=1,
        seas_s=12,
        grid_windows="3,5,6",
        grid_hiddens="16,32,124",
        lstm_lr=0.001,
        lstm_bs=32,
        lstm_epochs=200,
        lstm_patience=10,
        horizon=3,
    )
    assert isinstance(result, tuple)
    assert (
        len(result) == 15
    )  # Adjust based on the expected number of return values from run_pipeline


def test_run_pipeline_invalid_data():
    invalid_data = pd.DataFrame({"invalid_col": [1, 2, 3]})
    with pytest.raises(ValueError):
        run_pipeline(
            df_raw=invalid_data,
            date_col_name="date",
            value_col_name="value",
            do_outlier_iqr=True,
            waktu_col_name="waktu_kejadian",
            tkp_col_name="tkp",
            jenis_col_name="jenis_kejahatan",
            jumlah_col_name="jumlah_kejadian",
            coords_file=None,
            geojson_file=None,
            test_len=3,
            use_auto_arima=False,
            order_p=1,
            order_d=1,
            order_q=1,
            seas_P=1,
            seas_D=1,
            seas_Q=1,
            seas_s=12,
            grid_windows="3,5,6",
            grid_hiddens="16,32,124",
            lstm_lr=0.001,
            lstm_bs=32,
            lstm_epochs=200,
            lstm_patience=10,
            horizon=3,
        )
