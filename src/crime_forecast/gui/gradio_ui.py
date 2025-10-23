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
# div[role="tablist"] {
# 	display: block !important;
# }
# div[class="tab-wrapper svelte-i00v67"] {
# 	display: block !important;
# }
# .contain div[class="tab-wrapper svelte-i00v67"] {
# 	display: block !important;
# }
# .tab-container.svelte-i00v67.svelte-i00v67 {
# 	display: block !important;
# 	height: 0;
# 	padding-bottom: 0;
# }
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

    return (
        fig_main, fig_eda, fig_acf, fig_season, fig_calendar,
        metrics, comp_df,
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
        error_fig = make_info_fig(f"Terjadi Kesalahan:\n{e}", figsize=(10, 4))
        error_df = pd.DataFrame({"Error": [str(e)]})
        # Mengembalikan tuple dengan ukuran yang benar untuk semua output
        num_outputs = 14 # Sesuaikan jumlah ini jika output berubah
        
        # Buat daftar None dengan ukuran yang benar
        outputs = [None] * num_outputs
        
        # Assign error figure to plot outputs
        plot_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10] # Indeks plot
        for i in plot_indices:
            outputs[i] = error_fig
        
        # Assign error dataframe to table outputs
        df_indices = [5, 6, 13] # Indeks tabel
        for i in df_indices:
            outputs[i] = error_df

        return tuple(outputs)

with gr.Blocks(title=APP_TITLE, theme=theme, css=CUSTOM_CSS) as demo:

    # with gr.Sidebar():  
    #     gr.Markdown(
    #         f"""
    #         <div class='app-header'>
    #             <div style='font-size:28px; font-weight:800;'>{APP_TITLE}</div>
    #         </div>
    #         """
    #     )

    #     menu_home = gr.Button("Home", variant = "transparent")
    #     menu_training = gr.Button("Training & Results", variant = "transparent")
    #     menu_dashboard = gr.Button("Dashboard", variant="transparent")

    with gr.Tabs() as main_tabs:
        with gr.TabItem("Dashboard"):
            gr.Markdown(
                "Unggah CSV/XLSX atau isi tabel manual. Minimal `date` & `value` untuk forecasting. "
                "Kolom **opsional** untuk analisis kejadian: `waktu_kejadian`, `tkp`, `jenis_kejahatan`, `jumlah_kejadian`.\n\n"
                "Untuk **peta**: unggah **File Koordinat TKP** (CSV/XLSX: `tkp,lat,lon`). "
                "Opsional: unggah **GeoJSON Batas** (kecamatan/kelurahan) untuk choropleth."
            )
            df_in = gr.Dataframe(
                    value=EXAMPLE_DATA,
                    headers=list(EXAMPLE_DATA.columns),
                    row_count=(5, "dynamic"),
                    col_count=(5, "dynamic"),
                    label="Atau input manual di sini",
                )
            file_in = gr.File(label="Unggah CSV/XLSX (data utama)")
            
                
            date_col = gr.State(value="date")
            value_col = gr.State(value="value")
            do_outlier = gr.State(value=True)
            waktu_col = gr.State(value="waktu_kejadian" )
            tkp_col   = gr.State(value="tkp" )
            jenis_col = gr.State(value="jenis_kejahatan" )
            jumlah_col= gr.State(value="jumlah_kejadian" )
            coords_file = gr.State(value=r"src/crime_forecast/lokasi_kejahatan_gowa_with_coords_20251007_220421 (1).csv")
            geojson_file = gr.State(value=r"src/crime_forecast/lokasi_kejahatan_gowa_POINTS_20251007_220421.geojson")

        with gr.TabItem("Analisis"):
            test_len = gr.State(value=4,)
            horizon = gr.State(value=12)

            use_auto = gr.State(value=True)
            p = gr.State(value=1,)
            d = gr.State(value=2,)
            q = gr.State(value=2,)
            P = gr.State(value=1,)
            D = gr.State(value=0,)
            Q = gr.State(value=3,)
            s = gr.State(value=12)

            grid_win = gr.State(value="3")
            grid_hid = gr.State(value="124")
            lr = gr.State(value=0.001)
            batch_size = gr.State(value=32,)
            epochs = gr.State(value=200,)
            patience = gr.State(value=13,)

            run_btn = gr.Button("▶️ Jalankan Training, Analisis, & Peta", variant="primary")

        with gr.TabItem("📊 Hasil & Unduhan"):
            plot_main = gr.Plot(label="Grafik Prediksi: Train/Test, SARIMA vs Hybrid")
            plot_eda = gr.Plot(label="Boxplot & Histogram (setelah outlier handling)")
            plot_acf_pacf = gr.Plot(label="ACF & PACF")
            plot_season = gr.Plot(label="Rata-rata Kasus per Bulan")
            plot_calendar = gr.Plot(label="Peta Panas Bulanan (Year × Month)")
            table_metrics = gr.Dataframe(label="Metrik Evaluasi", interactive=False)
            table_comp = gr.Dataframe(label="Perbandingan Prediksi (Test)", interactive=False)

        with gr.TabItem("📈 Analisis TKP & Jenis"):
            with gr.Row():
                plot_top_tkp = gr.Plot(label="Top 10 Lokasi (TKP) Terbanyak")
                plot_top_jenis = gr.Plot(label="Top 10 Jenis Kejahatan Terbanyak")

            with gr.Row():
                plot_perjenis = gr.Plot(label="Jumlah Kejahatan per Bulan (per Jenis)")
                plot_total_bln = gr.Plot(label="Jumlah Kejahatan per Bulan (Total)")

        with gr.TabItem("🗺️ Peta Lokasi & Prediksi Masa Depan"):
            gr.Markdown("Hasil prediksi masa depan akan ditampilkan di sini setelah analisis dijalankan. Anda dapat mengatur panjang prediksi menggunakan `Horizon Forecast` di tab 'Parameter & Training'.")

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
                test_len, use_auto, p, d, q, P, D, Q, s,
                grid_win, grid_hid, lr, batch_size, epochs, patience, horizon,
            ],
            outputs=[
                # Tab 3
                plot_main, plot_eda, plot_acf_pacf, plot_season, plot_calendar,
                table_metrics, table_comp,
                # Tab 4
                plot_top_tkp, plot_top_jenis, plot_perjenis, plot_total_bln,
                # Tab 5 (Peta)
                map_html_comp, pred_range_output, # Hasil future_pred_df langsung ke tabel ini
            ],
            show_progress=True,
        )