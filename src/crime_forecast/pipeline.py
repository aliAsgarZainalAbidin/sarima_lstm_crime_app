from pathlib import Path
import io
import json
import math
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from crime_forecast.utils.maps import make_map_html, read_coords_file
from crime_forecast.utils.plotting import counts_per_location_type, plot_counts_per_location_type

from .data.preprocess import ensure_monthly_sum, remove_outliers_iqr
from .utils.metrics import mape, rmse_formula, mae_formula, r2_formula
from .utils.plotting import make_info_fig, plot_top_counts
from .models.sarima import fit_sarima, sarima_forecast
from .models.lstm import fit_lstm_and_predict
from crime_forecast.models.persistence import save_hybrid

import geopandas as gpd


def infer_event_columns(df: pd.DataFrame, waktu_name, tkp_name, jenis_name, jumlah_name):
    # normalisasi: mapping nama->lower-strip
    norm = {c: str(c).strip().lower() for c in df.columns}

    def by_exact(*cands):
        targets = [c.strip().lower() for c in cands]
        for k, v in norm.items():
            if v in targets: return k
        return None

    def by_contains(*subs):
        subs = [s.strip().lower() for s in subs]
        for k, v in norm.items():
            if any(s in v for s in subs): return k
        return None

    # jika user isi textbox dan ada di df -> pakai
    waktu = waktu_name if (waktu_name in df.columns) else None
    tkp   = tkp_name   if (tkp_name   in df.columns) else None
    jenis = jenis_name if (jenis_name in df.columns) else None
    jumlah= jumlah_name if (jumlah_name in df.columns) else None

    # fallback exact
    waktu = waktu or by_exact("waktu_kejadian","waktu","tanggal","tgl","date","time")
    tkp   = tkp   or by_exact("tkp","lokasi","location","tempat")
    jenis = jenis or by_exact("jenis_kejahatan","jenis","crime_type","kategori")
    jumlah= jumlah or by_exact("jumlah_kejadian","jumlah","value","count","total")

    # fallback contains
    waktu = waktu or by_contains("waktu","tanggal","date","time")
    tkp   = tkp   or by_contains("tkp","lokasi","location","tempat")
    jenis = jenis or by_contains("jenis","kejahat","crime","kategori")
    jumlah= jumlah or by_contains("jumlah","count","total","nilai","kejadian")

    return waktu, tkp, jenis, jumlah


# ---------- helpers umum ----------
def infer_columns(df: pd.DataFrame):
    # untuk deret waktu agregat bulanan (date,value)
    date_candidates = [c for c in df.columns if str(c).strip().lower() in
                       ["tanggal","tgl","date","time","waktu","bulan","month","period","periode","waktu_kejadian"]]
    value_candidates = [c for c in df.columns if str(c).strip().lower() in
                        ["jumlah","nilai","value","count","kejahatan","kasus","cases","y","total","jumlah_kejadian"]]
    date_col = date_candidates[0] if date_candidates else None
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    value_col = None
    for c in value_candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            value_col = c; break
    if value_col is None and numeric_cols:
        value_col = numeric_cols[0]
    return date_col, value_col


def run_pipeline(
    df_raw,
    date_col_name, value_col_name, do_outlier_iqr: bool,
    # event columns (optional)
    waktu_col_name, tkp_col_name, jenis_col_name, jumlah_col_name,
    # map files (optional)
    coords_file, geojson_file,
    # train/test + model
    test_len: int, use_auto_arima: bool,
    order_p: int, order_d: int, order_q: int,
    seas_P: int, seas_D: int, seas_Q: int, seas_s: int,
    grid_windows: str, grid_hiddens: str,
    lstm_lr: float, lstm_bs: int, lstm_epochs: int, lstm_patience: int,
    horizon: int,
):
    # 0) Event columns (for TKP/Type analysis)
    waktu_c, tkp_c, jenis_c, jumlah_c = infer_event_columns(
        df_raw, waktu_col_name, tkp_col_name, jenis_col_name, jumlah_col_name
    )

    # 1) Monthly aggregated time series (date,value)
    dc, vc = infer_columns(df_raw)
    date_col = date_col_name or dc
    value_col = value_col_name or vc
    if date_col is None or value_col is None:
        raise ValueError("Cannot recognize date/value columns. Please fill in column names manually.")
    ts = ensure_monthly_sum(df_raw, date_col, value_col)

    # 2) Outlier (optional)
    ts_clean = remove_outliers_iqr(ts) if do_outlier_iqr else ts.copy()

    # 3) Split
    n = ts_clean.shape[0]
    test_len = int(test_len)
    test_len = max(0, min(test_len, max(0, n - 12)))
    train = ts_clean.iloc[:-test_len].copy() if test_len > 0 else ts_clean.copy()
    test = ts_clean.iloc[-test_len:].copy() if test_len > 0 else ts_clean.iloc[0:0].copy()

    # 4) Log-transform
    train_t = np.log1p(train["value"])
    test_t = np.log1p(test["value"]) if test_len > 0 else pd.Series(dtype=float)

    # 5) SARIMA (optional auto_arima)
    order, seasonal_order = (1, 2, 2), (1, 0, 3, 7)
    if use_auto_arima:
        try:
            from pmdarima import auto_arima
            aa = auto_arima(
                train_t, seasonal=True, m=12, start_p=0, start_q=0, max_p=3, max_q=3,
                start_P=0, start_Q=0, max_P=2, max_Q=2, d=None, D=None, trace=False,
                error_action="ignore", suppress_warnings=True, stepwise=True, information_criterion="aicc",
            )
            order, seasonal_order = aa.order, aa.seasonal_order
        except Exception:
            order, seasonal_order = (order_p, order_d, order_q), (seas_P, seas_D, seas_Q, seas_s)
    else:
        order, seasonal_order = (order_p, order_d, order_q), (seas_P, seas_D, seas_Q, seas_s)

    sarima_fit_t = fit_sarima(train_t, order=order, seasonal_order=seasonal_order)

    # forecast test (log)
    sarima_forecast_test_t = sarima_forecast(sarima_fit_t, steps=test_len) if test_len > 0 else pd.Series(dtype=float)
    if test_len > 0:
        sarima_forecast_test_t.index = test.index

    # residual train (log)
    sarima_fitted_train_t = sarima_fit_t.fittedvalues
    residuals_train_t = (train_t - sarima_fitted_train_t).fillna(0.0)

    # 6) Small grid LSTM (10% train validation)
    try:
        gw = [int(x.strip()) for x in grid_windows.split(",") if x.strip()]
    except Exception:
        gw = [3, 5, 6]
    try:
        gh = [int(x.strip()) for x in grid_hiddens.split(",") if x.strip()]
    except Exception:
        gh = [16, 32, 124]

    val_h = max(6, int(0.1 * len(train_t)))
    best_conf, best_mape = None, float("inf")
    for win in gw:
        for hid in gh:
            try:
                pred_res_val_t = fit_lstm_and_predict(
                    residuals_train_t.iloc[:-val_h], val_h,
                    window=win, hidden=hid, lr=lstm_lr/10, batch_size=lstm_bs,
                    epochs=min(300, lstm_epochs+100), patience=max(10, lstm_patience),
                )
                sarima_val_t = sarima_forecast(sarima_fit_t, steps=val_h)
                sarima_val_t.index = residuals_train_t.index[-val_h:]
                hybrid_val_t = sarima_val_t.values + pred_res_val_t
                y_true_val = np.expm1(train_t.iloc[-val_h:]).values
                y_pred_val = np.expm1(hybrid_val_t).clip(min=0)
                mape_val = mape(y_true_val, y_pred_val)
                if mape_val < best_mape:
                    best_mape, best_conf = mape_val, (win, hid)
            except Exception:
                pass
    if best_conf is None:
        best_conf = (6, 32)
    best_window, best_hidden = best_conf

    # retrain & predict residual for test
    pred_res_test_t = (
        fit_lstm_and_predict(
            residuals_train_t, test_len,
            window=best_window, hidden=best_hidden,
            lr=lstm_lr, batch_size=lstm_bs, epochs=lstm_epochs, patience=lstm_patience,
        ) if test_len > 0 else np.array([])
    )

    # 7) Rescale & evaluate
    sarima_pred = np.expm1(sarima_forecast_test_t.values).clip(min=0) if test_len > 0 else np.array([])
    hybrid_pred = (np.expm1(sarima_forecast_test_t.values + pred_res_test_t).clip(min=0)
                   if test_len > 0 else np.array([]))
    y_true_test = test["value"].values if test_len > 0 else np.array([])

    # 7b) Future forecast
    future_pred_df = pd.DataFrame()
    if horizon > 0:
        # Retrain on all data
        ts_all_t = np.log1p(ts_clean["value"])
        sarima_all_fit_t = fit_sarima(ts_all_t, order=order, seasonal_order=seasonal_order)
        residuals_all_t = (ts_all_t - sarima_all_fit_t.fittedvalues).fillna(0.0)
        
        # Predict residuals
        pred_res_future_t = fit_lstm_and_predict(
            residuals_all_t, horizon, window=best_window, hidden=best_hidden,
            lr=lstm_lr, batch_size=lstm_bs, epochs=lstm_epochs, patience=lstm_patience)
        
        # Combine and create dataframe
        sarima_future_t_vals = sarima_forecast(sarima_all_fit_t, steps=horizon).values
        hybrid_future_t = np.expm1(sarima_future_t_vals + pred_res_future_t).clip(min=0)
        
        # Generate correct future dates starting from the month after the last data point
        last_date = ts_clean.index.max()
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        
        # Buat dataframe dengan format yang diinginkan untuk UI
        future_pred_df = pd.DataFrame({
            "Bulan": future_dates.strftime('%B %Y'),
            "Prediksi Jumlah Kejahatan": [int(round(p)) for p in hybrid_future_t]
        })

    # 8) Main plot (prediction)
    fig_main, ax = plt.subplots(figsize=(11.8, 4.6))
    ax.plot(train.index, train["value"], label="Train")
    if test_len > 0:
        ax.plot(test.index, test["value"], label="Test")
        ax.plot(test.index, sarima_pred, label="SARIMA")
        ax.plot(test.index, hybrid_pred, label="Hybrid")
    ax.set_title(f"Monthly Prediction: SARIMA vs Hybrid — best (win, hidden) = {best_conf}")
    ax.set_xlabel("Year"); ax.set_ylabel("Number of Cases"); ax.legend(); ax.grid(True)
    plt.tight_layout()

    # 9) EDA: Box/Hist
    fig_eda, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(ts_clean["value"]); axes[0].set_title("Box Plot Data"); axes[0].set_ylabel("Number of Cases")
    axes[1].hist(ts_clean["value"], bins=20); axes[1].set_title("Histogram Data")
    axes[1].set_xlabel("Number of Cases"); axes[1].set_ylabel("Frequency")
    plt.tight_layout()

    # 10) ACF/PACF
    fig_acf, axes2 = plt.subplots(2, 1, figsize=(11, 7))
    plot_acf(ts_clean["value"], lags=min(24, len(ts_clean) - 1), ax=axes2[0])
    plot_pacf(ts_clean["value"], lags=min(24, len(ts_clean) - 1), ax=axes2[1])
    plt.tight_layout()

    # 11) Monthly seasonality + peaks
    ts_all = ts_clean.copy()
    ts_all["month"] = ts_all.index.month
    month_avg = ts_all.groupby("month")["value"].mean().sort_values(ascending=False)
    month_rank = month_avg.reset_index().rename(columns={"value": "avg_value"})
    month_rank["month_name"] = month_rank["month"].apply(lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
    peak_months = month_rank.head(3)

    fig_season, ax3 = plt.subplots(figsize=(10.5, 4))
    order_plot = month_rank.sort_values("avg_value")
    ax3.bar(order_plot["month_name"], order_plot["avg_value"])
    ax3.set_title("Average Cases per Month (Seasonal)")
    ax3.set_xlabel("Month"); ax3.set_ylabel("Average Cases")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()

    # 11b) Peta Panas Bulanan (Year × Month)
    ts_all["year"] = ts_all.index.year
    ts_all["month_num"] = ts_all.index.month
    pivot_cal = ts_all.pivot_table(values="value", index="year", columns="month_num", aggfunc="sum", fill_value=0)
    for m in range(1, 13):
        if m not in pivot_cal.columns: pivot_cal[m] = 0
    pivot_cal = pivot_cal[sorted(pivot_cal.columns)]
    fig_calendar, axcal = plt.subplots(figsize=(10.5, 4.5))
    im = axcal.imshow(pivot_cal.values, aspect="auto")
    axcal.set_xticks(np.arange(12))
    axcal.set_xticklabels([pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)])
    axcal.set_yticks(np.arange(len(pivot_cal.index)))
    axcal.set_yticklabels(pivot_cal.index)
    axcal.set_title("Peta Panas Bulanan (Year × Month) — Intensitas Kasus")
    plt.colorbar(im, ax=axcal, label="Jumlah Kasus")
    for i in range(pivot_cal.shape[0]):
        for j in range(12):
            axcal.text(j, i, int(pivot_cal.values[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()

    # 12) Location/Type analysis from original data (without filtering time first)
    fig_top_tkp = plot_top_counts(
        df_raw[tkp_c] if (tkp_c and tkp_c in df_raw.columns) else None,
        title="Top 10 Locations (TKP) Most Frequent", xlabel="Location (TKP)"
    )
    fig_top_jenis = plot_top_counts(
        df_raw[jenis_c] if (jenis_c and jenis_c in df_raw.columns) else None,
        title="Top 10 Types of Crime Most Frequent", xlabel="Type of Crime"
    )

    # Monthly per-type & total
    if waktu_c and (waktu_c in df_raw.columns) and jenis_c and (jenis_c in df_raw.columns):
        data_event = df_raw.copy()
        data_event[waktu_c] = pd.to_datetime(data_event[waktu_c], errors="coerce")
        data_event = data_event.dropna(subset=[waktu_c])
        data_event["bulan"] = data_event[waktu_c].dt.to_period("M")
        if (jumlah_c is not None) and (jumlah_c in data_event.columns) and pd.api.types.is_numeric_dtype(data_event[jumlah_c]):
            agg = data_event.groupby([data_event["bulan"], jenis_c])[jumlah_c].sum().reset_index()
            qty_col = jumlah_c
        else:
            agg = data_event.groupby([data_event["bulan"], jenis_c]).size().reset_index(name="jumlah_kejd")
            qty_col = "jumlah_kejd"
        pivot_kind = agg.pivot(index="bulan", columns=jenis_c, values=qty_col).fillna(0)

        if not pivot_kind.empty:
            fig_perjenis, ax_pj = plt.subplots(figsize=(12.5, 6))
            for col in pivot_kind.columns:
                ax_pj.plot(pivot_kind.index.astype(str), pivot_kind[col], marker="o", label=str(col))
            ax_pj.set_title("Number of Crimes per Month (per Type)")
            ax_pj.set_xlabel("Month"); ax_pj.set_ylabel("Number")
            plt.xticks(rotation=45); ax_pj.grid(True, linestyle="--", alpha=0.5)
            ax_pj.legend(title="Type", ncols=2, fontsize=8); plt.tight_layout()
            total_bln = pivot_kind.sum(axis=1)
            fig_total_bln, ax_tb = plt.subplots(figsize=(12.5, 4.8))
            ax_tb.plot(total_bln.index.astype(str), total_bln.values, marker="o")
            ax_tb.set_title("Number of Crimes per Month (Total)")
            ax_tb.set_xlabel("Month"); ax_tb.set_ylabel("Number")
            plt.xticks(rotation=45); ax_tb.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
        else:
            fig_perjenis = make_info_fig("Data per-type per-month is empty.")
            fig_total_bln = make_info_fig("Total monthly data is empty.")
    else:
        fig_perjenis = make_info_fig("Time/type columns not found; per-month graphs cannot be created.")
        fig_total_bln = make_info_fig("Time/type columns not found; total monthly graphs cannot be created.")

    # 13) Peta lokasi (Marker/HeatMap; Choropleth opsional)
    coords_df = read_coords_file(coords_file) if coords_file is not None else None
    geojson_path = geojson_file if geojson_file is not None else None
    # pass jenis/jumlah column names so map popups include per-type counts
    map_html = make_map_html(df_raw, tkp_c, coords_df=coords_df, geojson_path=geojson_path,
                             jenis_col=jenis_c, jumlah_col=jumlah_c)

    # 13b) Hitung jumlah per lokasi × jenis untuk peta/tab peta
    pivot_loc_kind = pd.DataFrame()
    fig_loc_kind = make_info_fig("Data TKP/jenis tidak memadai untuk ditampilkan pada peta.")
    try:
        if tkp_c and jenis_c and (tkp_c in df_raw.columns) and (jenis_c in df_raw.columns):
            pivot_loc_kind = counts_per_location_type(df_raw, lokasi_col=tkp_c, jenis_col=jenis_c, jumlah_col=jumlah_c)
            fig_loc_kind = plot_counts_per_location_type(pivot_loc_kind, top_n=12)
    except Exception:
        pivot_loc_kind = pd.DataFrame()
        fig_loc_kind = make_info_fig("Gagal membuat grafik TKP×Jenis.")

    # 14) Historical & forecast dataframe + CSV download
    hist_df = train.copy()
    if test_len > 0:
        hist_df = pd.concat([train, test])
    hist_df = hist_df.reset_index().rename(columns={"date": "date", "value": "actual"})

    comp_df = pd.DataFrame()
    if test_len > 0:
        comp_df = pd.DataFrame(
            {"date": test.index.strftime("%Y-%m-%d"),
             "Actual": y_true_test, "SARIMA": sarima_pred, "Hybrid": hybrid_pred}
        )

    out_buf = io.StringIO()
    to_concat = []
    if not comp_df.empty:
        to_concat.append(
            comp_df.rename(columns={"Hybrid": "value"})[["date", "value"]].assign(kind="test_hybrid_pred")
        )
    hist_csv = hist_df.copy()
    hist_csv["date"] = hist_csv["date"].dt.strftime("%Y-%m-%d")
    hist_csv = hist_csv.rename(columns={"actual": "value"}) if "actual" in hist_csv.columns else hist_csv
    hist_csv = hist_csv[["date", "value"]].assign(kind="historical")
    to_concat.append(hist_csv)
    comb = pd.concat(to_concat, ignore_index=True)
    comb.to_csv(out_buf, index=False)
    csv_bytes = out_buf.getvalue().encode("utf-8")

    # 15) ADF test
    try:
        adf_stat, pval, *_ = adfuller(ts_clean["value"].values)
        adf_info = {"adf_stat": float(adf_stat), "pvalue": float(pval)}
    except Exception:
        adf_info = {"adf_stat": None, "pvalue": None}

    metrics_tbl = pd.DataFrame({
        "SARIMA": {
            "RMSE": np.nan if test_len==0 else rmse_formula(y_true_test, sarima_pred),
            "MAE":  np.nan if test_len==0 else mae_formula(y_true_test, sarima_pred),
            "R²":   np.nan if test_len==0 else r2_formula(y_true_test, sarima_pred),
            "MAPE (%)": np.nan if test_len==0 else mape(y_true_test, sarima_pred),
        },
        "Hybrid_SARIMA_LSTM": {
            "RMSE": np.nan if test_len==0 else rmse_formula(y_true_test, hybrid_pred),
            "MAE":  np.nan if test_len==0 else mae_formula(y_true_test, hybrid_pred),
            "R²":   np.nan if test_len==0 else r2_formula(y_true_test, hybrid_pred),
            "MAPE (%)": np.nan if test_len==0 else mape(y_true_test, hybrid_pred),
        },
    }).T

    metrics_tbl_for_ui = metrics_tbl.copy()
    metrics_tbl_for_ui.index.name = "Model"
    metrics_tbl_for_ui = metrics_tbl_for_ui.reset_index()

    kpis = {
        "rmse": None if test_len==0 else float(metrics_tbl.loc["Hybrid_SARIMA_LSTM","RMSE"]),
        "mae":  None if test_len==0 else float(metrics_tbl.loc["Hybrid_SARIMA_LSTM","MAE"]),
        "mape": None if test_len==0 else float(metrics_tbl.loc["Hybrid_SARIMA_LSTM","MAPE (%)"]),
        "r2":   None if test_len==0 else float(metrics_tbl.loc["Hybrid_SARIMA_LSTM","R²"]),
    }

    # example placement (pseudo-code):
    # sarima_res = fitted SARIMAXResults
    # lstm_model = trained torch.nn.Module
    # scaler_obj = scaler used for LSTM residuals (e.g. MinMaxScaler or StandardScaler)
    #
    # assign available objects in this scope (use None for items not stored) and save artifacts:
    models_dir = Path("models/hybrid")
    sarima_res = sarima_fit_t  # use fitted SARIMA from earlier in the pipeline
    lstm_model = None          # fit_lstm_and_predict does not return the model here; set to None or provide the model if available
    scaler_obj = None          # no scaler object was created/stored in this pipeline; set to None or provide if available
    save_meta = save_hybrid(sarima_res, lstm_model, scaler=scaler_obj, models_dir=models_dir)
    # save_meta contains paths to sarima, lstm, scaler and metadata

    return (
        # Prediksi & EDA
        fig_main, fig_eda, fig_acf, fig_season, fig_calendar,
        # Analisis kejadian
        fig_top_tkp, fig_top_jenis, fig_perjenis, fig_total_bln,
        # tabel/unduh & kpi
        metrics_tbl_for_ui, hist_df, comp_df, csv_bytes, kpis, adf_info, peak_months,
        # peta: pivot & figure + html
        pivot_loc_kind, fig_loc_kind, map_html, future_pred_df,
    )