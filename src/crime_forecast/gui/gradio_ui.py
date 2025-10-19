from gradio.themes import Soft
import pandas as pd
import numpy as np
import time
import gradio as gr
from crime_forecast.pipeline import run_pipeline
from crime_forecast.utils.plotting import make_info_fig
from datetime import datetime, date
import matplotlib.pyplot as plt
import io
import tempfile
import os
from pathlib import Path
import importlib
from crime_forecast.models.persistence import load_metadata, load_sarima, load_scaler
from crime_forecast.predict import forecast_with_hybrid
from crime_forecast.data.preprocess import ensure_monthly_sum

from datetime import datetime, date

# helper: prefer gr.datetime if available, fallback to other Date/Datetime variants or Text
def make_month_picker(label, placeholder=None):
    # prefer exact attribute 'datetime' as requested
    DateComp = None
    if hasattr(gr, "datetime"):
        DateComp = gr.datetime
    elif hasattr(gr, "Datetime"):
        DateComp = gr.Datetime
    elif hasattr(gr, "DateTime"):
        DateComp = gr.DateTime
    elif hasattr(gr, "Date"):
        DateComp = gr.Date
    elif hasattr(gr, "DatePicker"):
        DateComp = gr.DatePicker
    else:
        comps = getattr(gr, "components", None)
        if comps is not None:
            for name in ("datetime", "Datetime", "DateTime", "Date", "DatePicker"):
                if hasattr(comps, name):
                    DateComp = getattr(comps, name)
                    break

    if DateComp is not None:
        try:
            return DateComp(label=label)
        except Exception:
            try:
                return DateComp()
            except Exception:
                pass
    # fallback: plain text input (format YYYY-MM or YYYY-MM-DD)
    return gr.Text(label=label, placeholder=placeholder or "YYYY-MM or YYYY-MM-DD")


APP_TITLE = "Hybrid SARIMA–LSTM | Prediksi Kejahatan"
APP_TAGLINE = (
    "Pra-pemrosesan bulanan + pembersihan outlier, EDA (ADF/ACF/PACF), "
    "pelatihan SARIMA + LSTM residual, evaluasi, analisis TKP/Jenis, kalender musiman, dan peta lokasi."
)

theme = Soft(primary_hue="blue", secondary_hue="slate").set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_shadow="*shadow_drop_lg",
    block_radius="24px",
    button_large_padding="18px",
    button_large_radius="18px",
    input_radius="14px",
)

CUSTOM_CSS = """
.app-header {display:flex; align-items:center; gap:14px; padding: 8px 0 4px 0;}
.badge {font-size:12px; padding:4px 10px; border-radius:30px; background:#eef2ff; color:#3730a3;}
.subtle {color:#475569;}
.footer { color:#64748b; font-size:12px; text-align:center; padding: 14px 0 6px 0; }
hr.sep { border:0; height:1px; background:#e2e8f0; margin: 6px 0 10px 0; }
"""

EXAMPLE_DATA = pd.DataFrame({
    "date": pd.date_range("2019-01-01", periods=36, freq="MS"),
    "value": np.random.poisson(lam=120, size=36),
    "waktu_kejadian": pd.date_range("2019-01-10", periods=36, freq="MS"),
    "tkp": np.random.choice(["Pattallassang","Somba Opu","Pallangga","Bajeng"], size=36),
    "jenis_kejahatan": np.random.choice(["Pencurian","Penganiayaan","Curanmor"], size=36),
})

def _collect_df(file, table):
    if file is not None:
        try:
            if str(file.name).lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(file.name)
            else:
                return pd.read_csv(file.name)
        except Exception:
            return pd.read_excel(file.name)
    return pd.DataFrame(table)

def run_all(
    file, table,
    date_c, value_c, do_out,
    waktu_c, tkp_c, jenis_c, jumlah_c,
    coords_f, geojson_f,
    test_len, use_auto, p,d,q, P,D,Q, s,
    grid_win, grid_hid, lr, batch_size, epochs, patience, horizon,
):
    df_raw = _collect_df(file, table)

    # panggil pipeline — unpack dengan try/except untuk kompatibilitas
    try:
        (
            fig_main, fig_eda, fig_acf, fig_season, fig_calendar,
            fig_top_tkp, fig_top_jenis, fig_perjenis, fig_total_bln,
            metrics, hist_df, comp_df, csv_bytes, kpis, adf_info, peak_months,
            pivot_loc_kind, fig_loc_kind, map_html,
        ) = run_pipeline(
            df_raw,
            date_c, value_c, bool(do_out),
            waktu_c, tkp_c, jenis_c, jumlah_c,
            coords_f, geojson_f,
            int(test_len), bool(use_auto),
            int(p), int(d), int(q), int(P), int(D), int(Q), int(s),
            grid_win, grid_hid, float(lr), int(batch_size), int(epochs), int(patience),
            int(horizon),
        )
    except ValueError:
        # fallback jika run_pipeline versi lama tidak mengembalikan pivot/fig
        (
            fig_main, fig_eda, fig_acf, fig_season, fig_calendar,
            fig_top_tkp, fig_top_jenis, fig_perjenis, fig_total_bln,
            metrics, hist_df, comp_df, csv_bytes, peak_months, map_html
        ) = run_pipeline(
            df_raw,
            date_c, value_c, bool(do_out),
            waktu_c, tkp_c, jenis_c, jumlah_c,
            coords_f, geojson_f,
            int(test_len), bool(use_auto),
            int(p), int(d), int(q), int(P), int(D), int(Q), int(s),
            grid_win, grid_hid, float(lr), int(batch_size), int(epochs), int(patience),
            int(horizon),
        )
        kpis = {}
        adf_info = {}
        pivot_loc_kind = pd.DataFrame()
        fig_loc_kind = make_info_fig("Data TKP/jenis tidak tersedia.")

    # siapkan file unduhan dari csv_bytes
    tsname = int(time.time())
    fname = f"prediksi_hybrid_{tsname}.csv"
    with open(fname, "wb") as f:
        f.write(csv_bytes)

    rmse_v = (kpis or {}).get("rmse")
    mae_v  = (kpis or {}).get("mae")
    mape_v = (kpis or {}).get("mape")
    r2_v   = (kpis or {}).get("r2")
    adf_s = (adf_info or {}).get("adf_stat")
    adf_p = (adf_info or {}).get("pvalue")

    return (
        # Hasil & Unduhan
        fig_main, fig_eda, fig_acf, fig_season, fig_calendar,
        metrics, hist_df, comp_df, fname,
        rmse_v, mae_v, mape_v, r2_v, adf_s, adf_p, peak_months,
        # Analisis Kejadian
        fig_top_tkp, fig_top_jenis, fig_perjenis, fig_total_bln,
        # Peta: plot & tabel & html
        fig_loc_kind, table_loc_type, map_html,
    )

with gr.Blocks(title=APP_TITLE, theme=theme, css=CUSTOM_CSS) as demo:
    gr.Markdown(
        f"""
        <div class='app-header'>
          <div style='font-size:28px; font-weight:800;'>🚓 {APP_TITLE}</div>
          <span class='badge'>SARIMA + LSTM Residual</span>
        </div>
        <div class='subtle'>{APP_TAGLINE}</div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("📥 Data & Pra-Pemrosesan"):
            gr.Markdown(
                "Unggah CSV/XLSX atau isi tabel manual. Minimal `date` & `value` untuk forecasting. "
                "Kolom **opsional** untuk analisis kejadian: `waktu_kejadian`, `tkp`, `jenis_kejahatan`, `jumlah_kejadian`.\n\n"
                "Untuk **peta**: unggah **File Koordinat TKP** (CSV/XLSX: `tkp,lat,lon`). "
                "Opsional: unggah **GeoJSON Batas** (kecamatan/kelurahan) untuk choropleth."
            )
            with gr.Row():
                file_in = gr.File(label="Unggah CSV/XLSX (data utama)")
                df_in = gr.Dataframe(
                    value=EXAMPLE_DATA,
                    headers=list(EXAMPLE_DATA.columns),
                    row_count=(5, "dynamic"),
                    col_count=(5, "dynamic"),
                    label="Atau input manual di sini",
                )
            with gr.Row():
                date_col = gr.Textbox(value="date", label="Kolom Tanggal (forecast) (opsional)")
                value_col = gr.Textbox(value="value", label="Kolom Nilai (forecast) (opsional)")
                do_outlier = gr.Checkbox(value=True, label="Hapus Outlier (IQR)")
            with gr.Row():
                waktu_col = gr.Textbox(value="waktu_kejadian", label="Kolom Waktu Kejadian (opsional)")
                tkp_col   = gr.Textbox(value="tkp", label="Kolom TKP/Lokasi (opsional)")
                jenis_col = gr.Textbox(value="jenis_kejahatan", label="Kolom Jenis Kejahatan (opsional)")
                jumlah_col= gr.Textbox(value="jumlah_kejadian", label="Kolom Jumlah Kejadian (opsional)")
            with gr.Row():
                coords_file = gr.File(label="File Koordinat TKP (CSV/XLSX): tkp, lat, lon (opsional)", value="src\crime_forecast\lokasi_kejahatan_gowa_with_coords_20251007_220421 (1).csv")
                geojson_file = gr.File(label="GeoJSON Batas Kecamatan/Kelurahan (opsional)", value="src\crime_forecast\lokasi_kejahatan_gowa_POINTS_20251007_220421.geojson")

        with gr.TabItem("🧠 Parameter & Training"):
            with gr.Row():
                test_len = gr.Slider(0, 36, value=4, step=1, label="Panjang Test (bulan)")
                horizon = gr.Slider(1, 36, value=12, step=1, label="Horizon Forecast (bulan)")

            gr.Markdown("**SARIMA**")
            with gr.Row():
                use_auto = gr.Checkbox(value=True, label="Gunakan auto_arima (pmdarima)")
                p = gr.Slider(0, 5, value=1, step=1, label="p")
                d = gr.Slider(0, 2, value=2, step=1, label="d")
                q = gr.Slider(0, 5, value=2, step=1, label="q")
            with gr.Row():
                P = gr.Slider(0, 3, value=1, step=1, label="P")
                D = gr.Slider(0, 2, value=0, step=1, label="D")
                Q = gr.Slider(0, 4, value=3, step=1, label="Q")
                s = gr.Slider(0, 24, value=12, step=1, label="s (musim)")

            gr.Markdown("**LSTM Residual (Grid Kecil)**")
            with gr.Row():
                grid_win = gr.Textbox(value="3,5,6", label="Coba Window (comma-separated)")
                grid_hid = gr.Textbox(value="16,32,124", label="Coba Hidden Units (comma-separated)")
            with gr.Row():
                lr = gr.Number(value=0.001, label="Learning Rate")
                batch_size = gr.Slider(8, 128, value=32, step=1, label="Batch Size")
                epochs = gr.Slider(50, 1000, value=200, step=10, label="Epochs")
                patience = gr.Slider(3, 50, value=13, step=1, label="Patience")

            run_btn = gr.Button("▶️ Jalankan Training, Analisis, & Peta", variant="primary")

        with gr.TabItem("📊 Hasil & Unduhan"):
            plot_main = gr.Plot(label="Grafik Prediksi: Train/Test, SARIMA vs Hybrid")
            plot_eda = gr.Plot(label="Boxplot & Histogram (setelah outlier handling)")
            plot_acf_pacf = gr.Plot(label="ACF & PACF")
            plot_season = gr.Plot(label="Rata-rata Kasus per Bulan")
            plot_calendar = gr.Plot(label="Peta Panas Bulanan (Year × Month)")
            table_metrics = gr.Dataframe(label="Metrik Evaluasi", interactive=False)
            table_hist = gr.Dataframe(label="Data Historis (setelah praproses)", interactive=False)
            table_comp = gr.Dataframe(label="Perbandingan Prediksi (Test)", interactive=False)
            download = gr.File(label="Unduh CSV (historis + prediksi)")

            # KPI outputs (numbers) — harus didefinisikan sebelum dipakai di run_btn.click
            rmse_out = gr.Number(label="RMSE", interactive=False)
            mae_out = gr.Number(label="MAE", interactive=False)
            mape_out = gr.Number(label="MAPE (%)", interactive=False)
            r2_out = gr.Number(label="R²", interactive=False)

        with gr.TabItem("📈 Analisis TKP & Jenis"):
            plot_top_tkp = gr.Plot(label="Top 10 Lokasi (TKP) Terbanyak")
            plot_top_jenis = gr.Plot(label="Top 10 Jenis Kejahatan Terbanyak")
            plot_perjenis = gr.Plot(label="Jumlah Kejahatan per Bulan (per Jenis)")
            plot_total_bln = gr.Plot(label="Jumlah Kejahatan per Bulan (Total)")

        with gr.TabItem("🗺️ Peta Lokasi"):
            # map html (existing)
            map_html_comp = gr.HTML(value="<div style='padding:12px'>Peta akan tampil di sini setelah dijalankan.</div>")
            # tambahan: plot stacked per-lokasi per-jenis dan tabel pivot
            plot_loc_type = gr.Plot(label="Jumlah Kejahatan per Jenis di Lokasi (Top lokasi)")
            table_loc_type = gr.Dataframe(label="Tabel: Jumlah per Lokasi × Jenis", interactive=False)

        with gr.TabItem("🔮 Prediksi Bulanan"):
            # allow user to point to saved hybrid model folder (default models/hybrid)
            models_dir = gr.Text(value="models/hybrid", label="Folder model (metadata.json di dalam folder)")
            start_month = make_month_picker("Bulan/Tahun Awal (pilih hari pada bulan yang diinginkan)")
            end_month = make_month_picker("Bulan/Tahun Akhir (pilih hari pada bulan yang diinginkan)")
            predict_btn = gr.Button("Jalankan Prediksi untuk Rentang Bulan")

            plot_forecast = gr.Plot(label="Grafik Prediksi (rentang terpilih)")
            download_forecast = gr.File(label="Unduh CSV Prediksi (rentang terpilih)")

        # Diagnostik outputs
        adf_stat = gr.Number(label="ADF Statistic", interactive=False)
        adf_pval = gr.Number(label="ADF p-value", interactive=False)
        peak_tbl = gr.Dataframe(label="3 Bulan Puncak (Rata-rata Tertinggi)", interactive=False)

        run_btn.click(
            fn=run_all,
            inputs=[
                file_in, df_in,
                date_col, value_col, do_outlier,
                waktu_col, tkp_col, jenis_col, jumlah_col,
                coords_file, geojson_file,
                test_len, use_auto, p, d, q, P, D, Q, s,
                grid_win, grid_hid, lr, batch_size, epochs, patience, horizon,
            ],
            outputs=[
                # Tab 3
                plot_main, plot_eda, plot_acf_pacf, plot_season, plot_calendar,
                table_metrics, table_hist, table_comp, download,
                rmse_out, mae_out, mape_out, r2_out, adf_stat, adf_pval, peak_tbl,
                # Tab 4
                plot_top_tkp, plot_top_jenis, plot_perjenis, plot_total_bln,
                # Tab 5 (Peta) -> sekarang menerima plot, tabel, dan HTML
                plot_loc_type, table_loc_type, map_html_comp,
            ],
            show_progress=True,
        )

    # fungsi pembantu untuk hanya menjalankan prediksi dengan horizon tertentu
    def run_forecast(
        file, table,
        date_c, value_c, do_out,
        waktu_c, tkp_c, jenis_c, jumlah_c,
        coords_f, geojson_f,
        models_dir, start_month, end_month,
        use_auto, p,d,q, P,D,Q, s,
        grid_win, grid_hid, lr, batch_size, epochs, patience,
    ):
        df_raw = _collect_df(file, table)

        # parse start/end into date objects (reuse helper in this module)
        def parse_month_input(x):
            if x is None:
                return None
            if isinstance(x, (date, datetime)):
                return x if isinstance(x, date) else x.date()
            s = str(x).strip()
            for fmt in ("%Y-%m-%d", "%Y-%m", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    return datetime.strptime(s, fmt).date()
                except Exception:
                    continue
            try:
                dt = pd.to_datetime(s, errors="coerce")
                if pd.notna(dt):
                    return dt.date()
            except Exception:
                pass
            return None

        start = parse_month_input(start_month)
        end = parse_month_input(end_month)
        if start is None or end is None:
            return make_info_fig("Input tanggal tidak valid. Gunakan komponen tanggal atau format YYYY-MM / YYYY-MM-DD."), ""

        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        months = max(1, months)
        horizon = int(months)

        # load metadata from models_dir
        meta_path = Path(models_dir) / "metadata.json"
        sarima_path = lstm_path = scaler_path = None
        if meta_path.exists():
            try:
                meta = load_metadata(str(meta_path))
                sarima_path = meta.get("sarima")
                lstm_path = meta.get("lstm")
                scaler_path = meta.get("scaler")
            except Exception:
                sarima_path = lstm_path = scaler_path = None

        # ensure monthly historical series for forecasting
        try:
            ts_df = ensure_monthly_sum(df_raw, date_c, value_c)  # returns Series-like DataFrame indexed by date with 'value'
            hist_series = ts_df["value"] if isinstance(ts_df, pd.DataFrame) and "value" in ts_df.columns else pd.Series(ts_df)
        except Exception:
            hist_series = None

        # attempt hybrid forecast if both sarima and lstm state exist and LSTM class can be imported
        hybrid_result = None
        if sarima_path and Path(sarima_path).exists():
            # try to find LSTM class in a few common locations
            lstm_class = None
            if lstm_path and Path(lstm_path).exists():
                candidates = [
                    "crime_forecast.models.lstm.LSTMResidual",
                    "crime_forecast.models.lstm.LSTMModel",
                    "crime_forecast.models.lstm.LSTM",
                    "crime_forecast.lstm.LSTMModel",
                    "crime_forecast.models.arch.LSTM",
                ]
                for cand in candidates:
                    try:
                        modname, clsname = cand.rsplit(".", 1)
                        mod = importlib.import_module(modname)
                        lstm_class = getattr(mod, clsname)
                        break
                    except Exception:
                        lstm_class = None
                # if we found a class and have hist_series, call forecast_with_hybrid
                if lstm_class is not None and hist_series is not None:
                    try:
                        hybrid_result = forecast_with_hybrid(
                            sarima_path, lstm_class, lstm_path, scaler_path,
                            hist_series, start, end, device="cpu", window=12
                        )
                    except Exception:
                        hybrid_result = None

        # if hybrid failed, fallback to SARIMA-only forecast
        if hybrid_result is None:
            try:
                sarima = load_sarima(sarima_path) if sarima_path and Path(sarima_path).exists() else None
                if sarima is None:
                    return make_info_fig("Model SARIMA tidak ditemukan di folder yang diberikan."), ""
                # produce sarima forecast for horizon
                try:
                    sarima_pred = sarima.get_forecast(steps=horizon)
                    sarima_mean = pd.Series(sarima_pred.predicted_mean.values,
                                            index=pd.date_range(pd.to_datetime(start), periods=horizon, freq="MS"))
                except Exception:
                    last = pd.to_datetime(start) - pd.offsets.MonthBegin(1)
                    future_idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
                    sarima_mean = pd.Series([float(0)]*horizon, index=future_idx)
                result = pd.DataFrame({"date": sarima_mean.index, "predicted": sarima_mean.values})
                # limit to requested start..end
                mask = (result["date"] >= pd.to_datetime(start)) & (result["date"] <= pd.to_datetime(end))
                out_df = result.loc[mask].reset_index(drop=True)
            except Exception:
                return make_info_fig("Gagal melakukan prediksi menggunakan model yang tersimpan."), ""
        else:
            out_df = hybrid_result.copy()

        if out_df is None or out_df.empty:
            return make_info_fig("Tidak ada prediksi dalam rentang yang diberikan."), ""

        # plot
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(out_df["date"], out_df["predicted"], marker="o", linestyle="-")
        ax.set_title(f"Prediksi Jumlah Kejadian ({start.strftime('%Y-%m')} → {end.strftime('%Y-%m')})")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Prediksi Jumlah Kejadian")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        # save csv for download — gunakan temp dir dan pastikan kita mengembalikan path file yang valid
        fname = None
        try:
            fname = _safe_write_df_to_temp(out_df, filename_prefix=f"prediksi_rentang_{start.strftime('%Y%m')}_{end.strftime('%Y%m')}")
        except Exception:
            fname = None

        # Pastikan mengembalikan None jika tidak ada file valid (Gradio tidak boleh menerima direktori)
        return fig, fname

    # connect predict button
    predict_btn.click(
        fn=run_forecast,
        inputs=[
            file_in, df_in,
            date_col, value_col, do_outlier,
            waktu_col, tkp_col, jenis_col, jumlah_col,
            coords_file, geojson_file,
            models_dir, start_month, end_month,
            use_auto, p, d, q, P, D, Q, s,
            grid_win, grid_hid, lr, batch_size, epochs, patience,
        ],
        outputs=[plot_forecast, download_forecast],
        show_progress=True,
    )