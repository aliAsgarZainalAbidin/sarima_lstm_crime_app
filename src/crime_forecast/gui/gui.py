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


APP_TITLE = "Crime Prediction System"
APP_TAGLINE = (
    "Pra-pemrosesan bulanan + pembersihan outlier, EDA (ADF/ACF/PACF), "
    "pelatihan SARIMA + LSTM residual, evaluasi, analisis TKP/Jenis, kalender musiman, dan peta lokasi."
)

theme = Soft(primary_hue="blue", secondary_hue="slate").set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_shadow="*shadow_drop_lg",
    # block_radius="24px",
    button_large_padding="18px",
    button_large_radius="18px",
    input_radius="14px",
)

CUSTOM_CSS = """
.app-header {
	display: flex;
	align-items: center;
	gap: 14px;
	padding: 8px 0 4px 0;
}
.badge {
	font-size: 12px;
	padding: 4px 10px;
	border-radius: 30px;
	background: #eef2ff;
	color: #3730a3;
}
.subtle {
	color: #475569;
}
.footer {
	color: #64748b;
	font-size: 12px;
	text-align: center;
	padding: 14px 0 6px 0;
}
hr.sep {
	border: 0;
	height: 1px;
	background: #e2e8f0;
	margin: 6px 0 10px 0;
}
div[role="tablist"] {
	display: block !important;
}
div[class="tab-wrapper svelte-i00v67"] {
	display: block !important;
}
.contain div[class="tab-wrapper svelte-i00v67"] {
	display: block !important;
}
.tab-container.svelte-i00v67.svelte-i00v67 {
	display: block !important;
	height: 0;
	padding-bottom: 0;
}
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
            _, _, map_html, future_pred_df, # pivot_loc_kind dan fig_loc_kind tidak digunakan di UI
        ) = results
    except (ValueError, TypeError):
        # Fallback untuk versi pipeline yang lebih lama
        return tuple([make_info_fig("Error: Pipeline output mismatch.") for _ in range(24)])
    
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
        map_html, future_pred_df,
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
        error_fig = make_info_fig(f"Terjadi Kesalahan:\n{e}", figsize=(10, 4)) # For plots
        error_html = f"<div style='color:red;'>Terjadi Kesalahan: {e}</div>"
        
        # Mengembalikan tuple dengan ukuran yang benar untuk semua output
        num_outputs = 22 # Total outputs from _process_pipeline_outputs
        
        # Buat daftar None dengan ukuran yang benar
        outputs = [None] * num_outputs
        
        # Assign error figure to plot outputs
        plot_indices = [0, 1, 2, 3, 4, 16, 17, 18, 19]
        for i in plot_indices:
            outputs[i] = error_fig

        
        # Assign error message to number outputs
        number_indices = [9, 10, 11, 12, 13, 14]
        for i in number_indices:
            outputs[i] = None # Or np.nan, depending on desired display
        
        # Assign error message to HTML output
        outputs[20] = error_html
        
        return tuple(outputs)

with gr.Blocks(title=APP_TITLE, theme=theme, css=CUSTOM_CSS) as demo:

    with gr.Sidebar():  
        gr.Markdown(
            f"""
            <div class='app-header'>
                <div style='font-size:28px; font-weight:800;'>{APP_TITLE}</div>
            </div>
            """
        )

        menu_home = gr.Button("Home", variant = "transparent")
        menu_training = gr.Button("Training & Results", variant = "transparent")

    with gr.Tabs() as main_tabs:
        with gr.TabItem("", id="home"):
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
            date_col = gr.State(value="date", )
            value_col = gr.State(value="value",)
            do_outlier = gr.State(value=True)
            waktu_col = gr.State(value="waktu_kejadian" )
            tkp_col   = gr.State(value="tkp" )
            jenis_col = gr.State(value="jenis_kejahatan" )
            jumlah_col= gr.State(value="jumlah_kejadian" )
            coords_file = gr.State(value="src\crime_forecast\lokasi_kejahatan_gowa_with_coords_20251007_220421 (1).csv")
            geojson_file = gr.State(value="src\crime_forecast\lokasi_kejahatan_gowa_POINTS_20251007_220421.geojson")

        with gr.TabItem("", id="analysis"):
            gr.Markdown("### Parameter Pelatihan")
            test_len_state = gr.State(value=15)
            horizon_state = gr.State(value=12)
            use_auto_state = gr.State(value=True)
            p_state = gr.State(value=2)
            d_state = gr.State(value=0)
            q_state = gr.State(value=3)
            P_state = gr.State(value=1)
            D_state = gr.State(value=0)
            Q_state = gr.State(value=4)
            s_state = gr.State(value=4)
            grid_win_state = gr.State(value="3")
            grid_hid_state = gr.State(value="124")
            lr_state = gr.State(value=0.0001)
            batch_size_state = gr.State(value=32)
            epochs_state = gr.State(value=200)
            patience_state = gr.State(value=13)

            gr.Markdown("Catatan: Parameter pelatihan di bawah ini sekarang diatur secara statis menggunakan `gr.State` dan tidak dapat diubah dari UI. Untuk mengubahnya, edit kode sumber.")

            with gr.Row():
                gr.Number(value=test_len_state.value, interactive=False, label="Panjang Test (bulan)")
                gr.Number(value=horizon_state.value, interactive=False, label="Horizon Forecast (bulan)")

            gr.Markdown("**SARIMA**")
            with gr.Row():
                gr.Checkbox(value=use_auto_state.value, interactive=False, label="Gunakan auto_arima (pmdarima)")
                gr.Number(value=p_state.value, interactive=False, label="p")
                gr.Number(value=d_state.value, interactive=False, label="d")
                gr.Number(value=q_state.value, interactive=False, label="q")
            with gr.Row():
                gr.Number(value=P_state.value, interactive=False, label="P")
                gr.Number(value=D_state.value, interactive=False, label="D")
                gr.Number(value=Q_state.value, interactive=False, label="Q")
                gr.Number(value=s_state.value, interactive=False, label="s (musim)")

            gr.Markdown("**LSTM Residual (Grid Kecil)**")
            with gr.Row():
                gr.Textbox(value=grid_win_state.value, interactive=False, label="Coba Window (comma-separated)")
                gr.Textbox(value=grid_hid_state.value, interactive=False, label="Coba Hidden Units (comma-separated)")
            with gr.Row():
                gr.Number(value=lr_state.value, interactive=False, label="Learning Rate")
                gr.Number(value=batch_size_state.value, interactive=False, label="Batch Size")
                gr.Number(value=epochs_state.value, interactive=False, label="Epochs")
                gr.Number(value=patience_state.value, interactive=False, label="Patience")

            run_btn = gr.Button("▶️ Jalankan Training, Analisis, & Peta", variant="primary")

            gr.Markdown("### Diagnostik & Metrik")
            with gr.Row():
                adf_stat = gr.Number(label="ADF Statistic", interactive=False)
                adf_pval = gr.Number(label="ADF p-value", interactive=False)
                peak_tbl = gr.Dataframe(label="3 Bulan Puncak (Rata-rata Tertinggi)", interactive=False, type="pandas")
            
            with gr.Row():
                rmse_out = gr.Number(label="RMSE", interactive=False)
                mae_out = gr.Number(label="MAE", interactive=False)
                mape_out = gr.Number(label="MAPE (%)", interactive=False)
                r2_out = gr.Number(label="R²", interactive=False)
            
            table_metrics = gr.Dataframe(label="Metrik Evaluasi", interactive=False)

            gr.Markdown("### Hasil Prediksi & Analisis")
            plot_main = gr.Plot(label="Grafik Prediksi: Train/Test, SARIMA vs Hybrid")
            plot_eda = gr.Plot(label="Boxplot & Histogram (setelah outlier handling)")
            plot_acf_pacf = gr.Plot(label="ACF & PACF")
            plot_season = gr.Plot(label="Rata-rata Kasus per Bulan")
            plot_calendar = gr.Plot(label="Peta Panas Bulanan (Year × Month)")
            table_hist = gr.Dataframe(label="Data Historis (setelah praproses)", interactive=False)
            table_comp = gr.Dataframe(label="Perbandingan Prediksi (Test)", interactive=False)
            download = gr.File(label="Unduh CSV (historis + prediksi)")

            with gr.Row():
                plot_top_tkp = gr.Plot(label="Top 10 Lokasi (TKP) Terbanyak")
                plot_top_jenis = gr.Plot(label="Top 10 Jenis Kejahatan Terbanyak")
            with gr.Row():
                plot_perjenis = gr.Plot(label="Jumlah Kejahatan per Bulan (per Jenis)")
                plot_total_bln = gr.Plot(label="Jumlah Kejahatan per Bulan (Total)")

            gr.Markdown("### Peta & Prediksi Masa Depan")
            with gr.Row():
                # map html (existing)
                map_html_comp = gr.HTML(value="<div style='padding:12px'>Peta akan tampil di sini setelah dijalankan.</div>")
                pred_range_output = gr.Dataframe(label="Hasil Prediksi Jumlah Kejahatan", interactive=False)

        run_btn.click(
            fn=run_analysis_pipeline,
            inputs=[
                file_in, df_in,
                date_col, value_col, do_outlier,
                waktu_col, tkp_col, jenis_col, jumlah_col,
                coords_file, geojson_file,
                test_len_state, use_auto_state, p_state, d_state, q_state, P_state, D_state, Q_state, s_state,
                grid_win_state, grid_hid_state, lr_state, batch_size_state, epochs_state, patience_state, horizon_state,
            ],
            outputs=[
                plot_main, plot_eda, plot_acf_pacf, plot_season, plot_calendar,
                table_metrics, table_hist, table_comp, download,
                rmse_out, mae_out, mape_out, r2_out, adf_stat, adf_pval, peak_tbl,
                plot_top_tkp, plot_top_jenis, plot_perjenis, plot_total_bln,
                map_html_comp, pred_range_output,
            ],
            show_progress=True,
        )
    
    menu_home.click(
        fn=lambda: gr.update(selected="home"),
        inputs=None,
        outputs=[main_tabs],
    )

    # Updated to point to the consolidated "training" tab
    menu_training.click( 
        fn=lambda: gr.update(selected="analysis"),
        inputs=None,
        outputs=[main_tabs],
    )