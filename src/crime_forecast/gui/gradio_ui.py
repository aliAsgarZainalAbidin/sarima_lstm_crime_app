from gradio.themes import Soft
import pandas as pd
import numpy as np
import time
import gradio as gr
from crime_forecast.pipeline import run_pipeline
from crime_forecast.utils.plotting import make_info_fig
from datetime import datetime, date

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

def _prepare_pipeline_inputs(
    file, table, date_c, value_c, do_out, waktu_c, tkp_c, jenis_c, jumlah_c,
    coords_f, geojson_f, test_len, use_auto, p, d, q, P, D, Q, s,
    grid_win, grid_hid, lr, batch_size, epochs, patience, horizon
):
    """Mengumpulkan dan membersihkan semua input dari UI untuk pipeline."""
    df_raw = _collect_df(file, table)
    if df_raw.empty:
        raise ValueError("Data input kosong. Silakan unggah file atau isi tabel.")

    pipeline_args = {
        "df_raw": df_raw,
        "date_col_name": date_c, "value_col_name": value_c, "do_outlier_iqr": bool(do_out),
        "waktu_col_name": waktu_c, "tkp_col_name": tkp_c, "jenis_col_name": jenis_c, "jumlah_col_name": jumlah_c,
        "coords_file": coords_f, "geojson_file": geojson_f,
        "test_len": int(test_len), "use_auto_arima": bool(use_auto),
        "order_p": int(p), "order_d": int(d), "order_q": int(q),
        "seas_P": int(P), "seas_D": int(D), "seas_Q": int(Q), "seas_s": int(s),
        "grid_windows": grid_win, "grid_hiddens": grid_hid,
        "lstm_lr": float(lr), "lstm_bs": int(batch_size), "lstm_epochs": int(epochs), "lstm_patience": int(patience),
        "horizon": int(horizon),
    }
    return pipeline_args

def _process_pipeline_outputs(results):
    """Memproses hasil dari pipeline untuk ditampilkan di UI Gradio."""
    # Unpack hasil dengan aman, menangani kemungkinan versi pipeline yang berbeda
    try:
        (
            fig_main, fig_eda, fig_acf, fig_season, fig_calendar,
            fig_top_tkp, fig_top_jenis, fig_perjenis, fig_total_bln,
            metrics, hist_df, comp_df, csv_bytes, kpis, adf_info, peak_months,
            _, _, map_html,  # pivot_loc_kind dan fig_loc_kind tidak digunakan di UI
        ) = results
    except (ValueError, TypeError):
        # Fallback untuk versi pipeline yang lebih lama
        (fig_main, fig_eda, fig_acf, fig_season, fig_calendar,
         fig_top_tkp, fig_top_jenis, fig_perjenis, fig_total_bln,
         metrics, hist_df, comp_df, csv_bytes, peak_months, map_html) = results
        kpis, adf_info = {}, {}

    # Siapkan file unduhan dari byte CSV
    tsname = int(time.time())
    fname = f"prediksi_hybrid_{tsname}.csv"
    with open(fname, "wb") as f:
        f.write(csv_bytes)

    # Ekstrak KPI dan statistik diagnostik dengan aman
    kpis = kpis or {}
    adf_info = adf_info or {}
    rmse_v, mae_v, mape_v, r2_v = kpis.get("rmse"), kpis.get("mae"), kpis.get("mape"), kpis.get("r2")
    adf_s, adf_p = adf_info.get("adf_stat"), adf_info.get("pvalue")

    return (
        fig_main, fig_eda, fig_acf, fig_season, fig_calendar,
        metrics, hist_df, comp_df, fname,
        rmse_v, mae_v, mape_v, r2_v, adf_s, adf_p, peak_months,
        fig_top_tkp, fig_top_jenis, fig_perjenis, fig_total_bln,
        map_html,
    )

def run_analysis_pipeline(
    file, table, date_c, value_c, do_out, waktu_c, tkp_c, jenis_c, jumlah_c,
    coords_f, geojson_f, test_len, use_auto, p, d, q, P, D, Q, s,
    grid_win, grid_hid, lr, batch_size, epochs, patience, horizon,
):
    """Fungsi utama yang dipanggil oleh Gradio untuk menjalankan seluruh alur kerja."""
    try:
        # 1. Kumpulkan dan siapkan semua input
        pipeline_args = _prepare_pipeline_inputs(
            file, table, date_c, value_c, do_out, waktu_c, tkp_c, jenis_c, jumlah_c,
            coords_f, geojson_f, test_len, use_auto, p, d, q, P, D, Q, s,
            grid_win, grid_hid, lr, batch_size, epochs, patience, horizon
        )

        # 2. Jalankan pipeline utama
        results = run_pipeline(**pipeline_args)

        # 3. Proses output untuk ditampilkan di UI
        return _process_pipeline_outputs(results)

    except Exception as e:
        # Menangani kesalahan dengan menampilkan pesan di beberapa output plot
        error_fig = make_info_fig(f"Terjadi Kesalahan:\n{e}", figsize=(10, 4))
        # Mengembalikan tuple dengan ukuran yang benar untuk semua output
        num_outputs = 22 # Sesuaikan jumlah ini jika output berubah
        return tuple([error_fig if i < 5 else None for i in range(num_outputs)])

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

        # Diagnostik outputs
        adf_stat = gr.Number(label="ADF Statistic", interactive=False)
        adf_pval = gr.Number(label="ADF p-value", interactive=False)
        peak_tbl = gr.Dataframe(label="3 Bulan Puncak (Rata-rata Tertinggi)", interactive=False)

        run_btn.click(
            fn=run_analysis_pipeline,
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
                # Tab 5 (Peta)
                map_html_comp,
            ],
            show_progress=True,
        )