from gradio.themes import Soft
import pandas as pd
import numpy as np
import gradio as gr
from .gradio_callbacks import (
    run_analysis_pipeline,
    save_new_event,
    import_data_to_db,
    load_data_from_db,
    update_prediction_range_text,
)


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

theme = Soft(
    primary_hue="blue",
    secondary_hue="blue",
    text_size="md",
    spacing_size="md",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"],
)

css = """
.gradio-container {
    overflow: visible !important;
    max-width: none !important;
}
#controls {
    max-height: 100vh;
    flex-wrap: unset;
    overflow-y: scroll;
    position: sticky;
    top: 0;
}
#controls::-webkit-scrollbar {
  -webkit-appearance: none;
  width: 7px;
}

#controls::-webkit-scrollbar-thumb {
  border-radius: 4px;
  background-color: rgba(0, 0, 0, .5);
  box-shadow: 0 0 1px rgba(255, 255, 255, .5);
}
"""

EXAMPLE_DATA = pd.DataFrame(
    {
        "date": pd.date_range("2019-01-01", periods=36, freq="MS"),
        "value": np.random.poisson(lam=120, size=36),
        "waktu_kejadian": pd.date_range("2019-01-10", periods=36, freq="MS"),
        "tkp": np.random.choice(
            ["Pattallassang", "Somba Opu", "Pallangga", "Bajeng"], size=36
        ),
        "jenis_kejahatan": np.random.choice(
            ["Pencurian", "Penganiayaan", "Curanmor"], size=36
        ),
    }
)

with gr.Blocks(title=APP_TITLE, theme=theme, css=css) as demo:
    with gr.Tabs() as main_tabs:
        with gr.TabItem("Input Data Kejadian"):
            with gr.Row(variant="compact", equal_height=False):
                with gr.Column(scale=1, elem_id="controls", min_width=400):
                    gr.Markdown("## Input Data Kejadian Baru")
                    input_waktu = gr.DateTime(label="Waktu Kejadian")
                    input_jenis = gr.Dropdown(
                        label="Jenis Kejahatan",
                        choices=[
                            "cabul",
                            "curanmor",
                            "curas",
                            "curat",
                            "kdrt",
                            "pencurian",
                            "penganiayaan",
                            "pengeroyokan",
                            "penggelapan",
                            "penipuan",
                        ],
                    )
                    input_tkp = gr.Dropdown(
                        label="Tempat Kejadian Perkara (TKP)",
                        choices=[
                            "bajeng",
                            "bajeng barat",
                            "barombong",
                            "biringbulu",
                            "bontolempangan",
                            "bontomarannu",
                            "bontonompo",
                            "bontonomposelatan",
                            "bungaya",
                            "manuju",
                            "pallangga",
                            "parangloe",
                            "parigi",
                            "pattallassang",
                            "sombaopu",
                            "tinggimoncong",
                            "tombolopao",
                            "tompobulu",
                        ],
                    )
                    save_button = gr.Button(
                        "Simpan",
                        variant="primary",
                    )

                with gr.Column(scale=6, elem_id="app"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            gr.Markdown("## Impor Data Kejadian dari File")
                        with gr.Column(scale=1, min_width=80):
                            import_file_input = gr.UploadButton(
                                "Import",
                                file_types=[".csv", ".xlsx"],
                                file_count="single",
                                icon="src/crime_forecast/assets/icons/upload.svg",
                                variant="secondary",
                            )

                    db_table_output = gr.Dataframe(
                        label="",
                        interactive=False,
                        show_label=False,
                        headers=[
                            "Waktu Kejadian",
                            "Jenis Kejadian",
                            "TKP",
                            "Jumlah Kejadian",
                        ],
                    )

                    date_col = gr.State(value="waktu_kejadian")
                    value_col = gr.State(value="value")
                    do_outlier = gr.State(value=True)
                    waktu_col = gr.State(value="waktu_kejadian")
                    tkp_col = gr.State(value="tkp")
                    jenis_col = gr.State(value="jenis_kejahatan")
                    jumlah_col = gr.State(value="jumlah_kejadian")
                    coords_file = gr.State(
                        value=r"src/crime_forecast/lokasi_kejahatan_gowa_with_coords_20251007_220421 (1).csv"
                    )
                    geojson_file = gr.State(
                        value=r"src/crime_forecast/lokasi_kejahatan_gowa_POINTS_20251007_220421.geojson"
                    )

        with gr.TabItem("Analisis"):
            with gr.Row():
                with gr.Column(scale=1, elem_id="controls", min_width=400):
                    test_len = gr.State(value=15)
                    horizon = gr.Number(
                        label="Jumlah Bulan Prediksi (bulan)", value=12, step=1, minimum=1
                    )
                    prediction_range_display = gr.Markdown(
                        "Rentang prediksi akan ditampilkan di sini setelah data dimuat."
                    )
                    end_date = gr.State(
                        value="June 2025",
                    )

                    use_auto = gr.State(value=True)
                    p = gr.State(
                        value=2,
                    )
                    d = gr.State(
                        value=0,
                    )
                    q = gr.State(
                        value=3,
                    )
                    P = gr.State(
                        value=1,
                    )
                    D = gr.State(
                        value=0,
                    )
                    Q = gr.State(
                        value=4,
                    )
                    s = gr.State(value=4)

                    grid_win = gr.State(value="6")
                    grid_hid = gr.State(value="124")
                    lr = gr.State(value=0.001)
                    batch_size = gr.State(
                        value=8,
                    )
                    epochs = gr.State(
                        value=200,
                    )
                    patience = gr.State(
                        value=13,
                    )

                    run_btn = gr.Button(
                        "▶️ Jalankan Training, Analisis, & Peta", variant="primary"
                    )

                with gr.Column(scale=6, elem_id="app"):
                    with gr.Accordion("🗺️ Peta Lokasi & Prediksi Masa Depan", open=False):
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=2):
                                gr.Markdown("## Peta Lokasi Kejahatan")
                                map_html_comp = gr.HTML(
                                    value="<div style='padding:12px'>Analisis sedang Berjalan</div>"
                                )
                            with gr.Column(scale=1, min_width=350):
                                gr.Markdown("## Hasil Prediksi Masa Depan")
                                pred_range_output = gr.Dataframe(
                                    label="Hasil Prediksi Jumlah Kejahatan",
                                    interactive=False,
                                    show_label=False,
                                )
                    with gr.Accordion("📊 Hasil ", open=False):
                        table_metrics = gr.Dataframe(label="Metrik Evaluasi", interactive=False)
                        table_comp = gr.Dataframe(
                            label="Perbandingan Prediksi (Test)", interactive=False
                        )
                        plot_main = gr.Plot(label="Grafik Prediksi: Train/Test, SARIMA vs Hybrid")
                        plot_season = gr.Plot(label="Rata-rata Kasus per Bulan")

                    with gr.Accordion("📈 Analisis TKP & Jenis", open=False):
                        plot_top_tkp = gr.Plot(label="Top 10 Lokasi (TKP) Terbanyak")
                        plot_top_jenis = gr.Plot(label="Top 10 Jenis Kejahatan Terbanyak")
                        plot_perjenis = gr.Plot(label="Jumlah Kejahatan per Bulan (per Jenis)")


        horizon.change(
            fn=update_prediction_range_text,
            inputs=[db_table_output, horizon],
            outputs=[prediction_range_display, end_date],
        )

        save_button.click(
            fn=save_new_event,
            inputs=[input_waktu, input_jenis, input_tkp],
        ).then(
            fn=load_data_from_db,
            inputs=None,
            outputs=[db_table_output],
        ).then(
            fn=update_prediction_range_text,
            inputs=[db_table_output, horizon],
            outputs=[prediction_range_display,end_date],
        ).then(
            fn=run_analysis_pipeline,
            inputs=[
                db_table_output,
                date_col,
                value_col,
                do_outlier,
                waktu_col,
                tkp_col,
                jenis_col,
                jumlah_col,
                coords_file,
                geojson_file,
                test_len,
                use_auto,
                p,
                d,
                q,
                P,
                D,
                Q,
                s,
                grid_win,
                grid_hid,
                lr,
                batch_size,
                epochs,
                patience,
                horizon,
                end_date
            ],
            outputs=[
                # Tab 3
                plot_main,
                plot_season,
                table_metrics,
                table_comp,
                # Tab 4
                plot_top_tkp,
                plot_top_jenis,
                plot_perjenis,
                # Tab 5 (Peta)
                map_html_comp,
                pred_range_output,  # Hasil future_pred_df langsung ke tabel ini
            ],
            show_progress=True,
        )

        import_file_input.upload(
            fn=import_data_to_db,
            inputs=[import_file_input],
        ).then(
            fn=load_data_from_db, inputs=None, outputs=[db_table_output]
        ).then(
            fn=update_prediction_range_text, inputs=[db_table_output, horizon], outputs=[prediction_range_display,end_date]
        ).then(
            fn=run_analysis_pipeline,
            inputs=[
                db_table_output,
                date_col,
                value_col,
                do_outlier,
                waktu_col,
                tkp_col,
                jenis_col,
                jumlah_col,
                coords_file,
                geojson_file,
                test_len,
                use_auto,
                p,
                d,
                q,
                P,
                D,
                Q,
                s,
                grid_win,
                grid_hid,
                lr,
                batch_size,
                epochs,
                patience,
                horizon,
                end_date
            ],
            outputs=[
                # Tab 3
                plot_main,
                plot_season,
                table_metrics,
                table_comp,
                # Tab 4
                plot_top_tkp,
                plot_top_jenis,
                plot_perjenis,
                # Tab 5 (Peta)
                map_html_comp,
                pred_range_output,  # Hasil future_pred_df langsung ke tabel ini
            ],
            show_progress=True,
        )

        run_btn.click(
            fn=run_analysis_pipeline,
            inputs=[
                db_table_output,
                date_col,
                value_col,
                do_outlier,
                waktu_col,
                tkp_col,
                jenis_col,
                jumlah_col,
                coords_file,
                geojson_file,
                test_len,
                use_auto,
                p,
                d,
                q,
                P,
                D,
                Q,
                s,
                grid_win,
                grid_hid,
                lr,
                batch_size,
                epochs,
                patience,
                horizon,
                end_date
            ],
            outputs=[
                # Tab 3
                plot_main,
                plot_season,
                table_metrics,
                table_comp,
                # Tab 4
                plot_top_tkp,
                plot_top_jenis,
                plot_perjenis,
                # Tab 5 (Peta)
                map_html_comp,
                pred_range_output,  # Hasil future_pred_df langsung ke tabel ini
            ],
            show_progress=True,
        )

    demo.load(
        load_data_from_db, inputs=None, outputs=[db_table_output]
    ).then(
        fn=update_prediction_range_text, inputs=[db_table_output, horizon], outputs=[prediction_range_display,end_date]
    ).then(
        fn=run_analysis_pipeline,
        inputs=[
            db_table_output,
            date_col,
            value_col,
            do_outlier,
            waktu_col,
            tkp_col,
            jenis_col,
            jumlah_col,
            coords_file,
            geojson_file,
            test_len,
            use_auto,
            p,
            d,
            q,
            P,
            D,
            Q,
            s,
            grid_win,
            grid_hid,
            lr,
            batch_size,
            epochs,
            patience,
            horizon,
            end_date
        ],
        outputs=[
            # Tab 3
            plot_main,
            plot_season,
            table_metrics,
            table_comp,
            # Tab 4
            plot_top_tkp,
            plot_top_jenis,
            plot_perjenis,
            # Tab 5 (Peta)
            map_html_comp,
            pred_range_output,  # Hasil future_pred_df langsung ke tabel ini
        ],
        show_progress=True,
    )
