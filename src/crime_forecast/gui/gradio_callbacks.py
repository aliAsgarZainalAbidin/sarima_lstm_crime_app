import pandas as pd
import time
import sqlite3
from pandas.tseries.offsets import DateOffset
from datetime import datetime
import gradio as gr

from crime_forecast.pipeline import run_pipeline
from crime_forecast.utils.plotting import make_info_fig


def use_db_data():
    """Memuat data dari database dan mengembalikannya untuk diisi ke dalam dataframe."""
    df = load_data_from_db()
    return df


def _collect_df_from_db(db_data):
    """Mengonversi output dataframe dari Gradio (yang bisa berupa dict) menjadi DataFrame pandas."""
    if isinstance(db_data, dict):  # Gradio < 4 passes dict for empty dataframe
        return pd.DataFrame(db_data.get("data", []), columns=db_data.get("headers", []))
    return pd.DataFrame(db_data)


def _prepare_pipeline_inputs(
    db_data,
    date_c,
    value_c,
    do_out,
    waktu_c,
    tkp_c,
    jenis_c,
    jumlah_c,
    coords_f,
    geojson_f,
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
):
    """Mengumpulkan dan membersihkan semua input dari UI untuk pipeline."""
    df_raw = _collect_df_from_db(db_data)
    if df_raw.empty:
        raise ValueError("Data input kosong. Silakan unggah file atau isi tabel.")

    pipeline_args = {
        "df_raw": df_raw,
        "date_col_name": date_c,
        "value_col_name": value_c,
        "do_outlier_iqr": bool(do_out),
        "waktu_col_name": waktu_c,
        "tkp_col_name": tkp_c,
        "jenis_col_name": jenis_c,
        "jumlah_col_name": jumlah_c,
        "coords_file": coords_f,
        "geojson_file": geojson_f,
        "test_len": int(test_len),
        "use_auto_arima": bool(use_auto),
        "order_p": int(p),
        "order_d": int(d),
        "order_q": int(q),
        "seas_P": int(P),
        "seas_D": int(D),
        "seas_Q": int(Q),
        "seas_s": int(s),
        "grid_windows": grid_win,
        "grid_hiddens": grid_hid,
        "lstm_lr": float(lr),
        "lstm_bs": int(batch_size),
        "lstm_epochs": int(epochs),
        "lstm_patience": int(patience),
        "horizon": int(horizon),
        "end_date": end_date,
    }
    return pipeline_args


def _process_pipeline_outputs(results):
    """Memproses hasil dari pipeline untuk ditampilkan di UI Gradio."""
    # Unpack hasil dengan aman, menangani kemungkinan versi pipeline yang berbeda
    try:
        (
            fig_main,
            fig_eda,
            fig_acf,
            fig_season,
            fig_calendar,
            fig_top_tkp,
            fig_top_jenis,
            fig_perjenis,
            metrics,
            hist_df,
            comp_df,
            csv_bytes,
            kpis,
            adf_info,
            peak_months,
            _,
            _,
            map_html,
            future_pred_df,  # pivot_loc_kind dan fig_loc_kind tidak digunakan di UI
        ) = results
    except (ValueError, TypeError):
        # Fallback untuk versi pipeline yang lebih lama
        return tuple(
            [make_info_fig("Error: Pipeline output mismatch.") for _ in range(24)]
        )

    # Siapkan file unduhan dari byte CSV
    # tsname = int(time.time())
    # fname = f"prediksi_hybrid_{tsname}.csv"
    # with open(fname, "wb") as f:
    #     f.write(csv_bytes)

    return (
        fig_main,
        fig_season,
        metrics,
        comp_df,
        map_html,
        future_pred_df,
    )


def update_horizon_choices(df_data):
    """
    Memperbarui pilihan dropdown horizon berdasarkan tanggal terakhir dalam data.
    Menghasilkan pilihan berupa (Label: "Bulan Tahun", Value: jumlah bulan).
    """
    try:
        df = _collect_df_from_db(df_data)
        if df.empty:
            return gr.DateTime(value=None, interactive=False)

        dates = pd.to_datetime(df.iloc[:, 0], errors="coerce").dropna()
        if dates.empty:
            return gr.DateTime(value=None, interactive=False)

        last_date = dates.max()
        # Default 12 bulan ke depan
        default_date = last_date + DateOffset(months=12)

        return gr.DateTime(value=default_date, interactive=True)

    except Exception:
        return gr.DateTime(value=None, interactive=False)


def update_prediction_range_text(horizon, df_data):
    """
    Memperbarui teks rentang prediksi berdasarkan tanggal terakhir dalam data dan horizon yang dipilih.
    """
    if not horizon:
        return "Pilih bulan akhir prediksi.", ""

    try:
        df = _collect_df_from_db(df_data)
        if df.empty:
            return "Data belum dimuat.", ""

        dates = pd.to_datetime(df.iloc[:, 0], errors="coerce").dropna()
        if dates.empty:
            return "Data tanggal tidak valid.", ""

        last_date = dates.max()
        start_pred_date = (last_date.to_period("M") + 1).to_timestamp()
        end_pred_date = pd.to_datetime(horizon)
        start_month_name = start_pred_date.strftime('%B %Y')

        return (
            f"Prediksi akan mencakup rentang dari **{start_month_name}** hingga **{end_pred_date.strftime('%B %Y')}**.",
            end_pred_date.strftime('%B %Y')
        )
    except Exception:
        return "Gagal menghitung rentang prediksi. Periksa format data tanggal."


def run_analysis_pipeline(
    db_data,
    date_c,
    value_c,
    do_out,
    waktu_c,
    tkp_c,
    jenis_c,
    jumlah_c,
    coords_f,
    geojson_f,
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
    end_date,
):
    """Fungsi utama yang dipanggil oleh Gradio untuk menjalankan seluruh alur kerja."""

    # Hitung integer horizon dari input tanggal
    horizon_int = 12
    if horizon is not None:
        try:
            df_temp = _collect_df_from_db(db_data)
            if not df_temp.empty:
                dates = pd.to_datetime(df_temp.iloc[:, 0], errors="coerce").dropna()
                if not dates.empty:
                    last_date = dates.max()
                    target_date = pd.to_datetime(horizon)
                    diff = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
                    horizon_int = max(1, int(diff))
        except Exception:
            pass
    else:
        gr.Warning("Bulan akhir prediksi tidak dipilih. Menggunakan default 12 bulan.")
 
    # 1. Kumpulkan dan siapkan semua input
    pipeline_args = _prepare_pipeline_inputs(
        db_data,
        date_c,
        value_c,
        do_out,
        waktu_c,
        tkp_c,
        jenis_c,
        jumlah_c,
        coords_f,
        geojson_f,
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
        horizon_int,
        end_date
    )

    # 2. Jalankan pipeline utama
    results = run_pipeline(**pipeline_args)

    # 3. Proses output untuk ditampilkan di UI
    return _process_pipeline_outputs(results)


def save_new_event(waktu, jenis, tkp, checkTabel=False):
    """Menyimpan data kejadian baru ke database SQLite lokal."""
    if (not waktu or not jenis or not tkp) and not checkTabel:
        gr.Warning("Semua field harus diisi.")
        return "Gagal: Semua field harus diisi."

    db_file = "database_kejadian.db"
    table_name = "kejadian"
    conn = None

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                waktu_kejadian TEXT NOT NULL,
                jenis_kejahatan TEXT NOT NULL,
                tkp TEXT NOT NULL,
                jumlah_kejadian INTEGER NOT NULL DEFAULT 1
            )
        """)
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        if "jumlah_kejadian" not in columns:
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN jumlah_kejadian INTEGER NOT NULL DEFAULT 1"
            )
            conn.commit()

        if isinstance(waktu, float):
            waktu_dt = datetime.fromtimestamp(waktu)
        else:
            waktu_dt = waktu
        waktu_str = waktu_dt.isoformat()

        cursor.execute(
            f"INSERT INTO {table_name} (waktu_kejadian, jenis_kejahatan, tkp) VALUES (?, ?, ?)",
            (waktu_str, jenis, tkp),
        )
        conn.commit()
        gr.Success(f"Data kejadian di '{tkp}' berhasil disimpan ke database.")
        return f"Sukses: Data kejadian di '{tkp}' berhasil disimpan ke database SQLite."
    except sqlite3.Error as e:
        gr.Error(f"Gagal menyimpan data ke SQLite: {e}")
        return f"Gagal menyimpan data ke SQLite: {e}"
    except Exception as e:
        gr.Error(f"Gagal menyimpan data: {e}")
        return f"Gagal menyimpan data: {e}"
    finally:
        if conn:
            conn.close()



def import_data_to_db(file):
    """Mengimpor data dari file Excel/CSV ke database SQLite."""
    if file is None:
        gr.Warning("Tidak ada file yang diunggah.")
        return "Gagal: Tidak ada file yang diunggah."

    try:
        file_path = file.name
        if file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        gr.Error(f"Gagal membaca file: {e}")
        return f"Gagal membaca file: {e}"

    df.columns = [str(col).strip().lower() for col in df.columns]
    required_cols = ["waktu_kejadian", "jenis_kejahatan", "tkp"]
    if any(col not in df.columns for col in required_cols):
        gr.Error(
            f"Kolom yang dibutuhkan ({', '.join(required_cols)}) tidak ada di file."
        )
        return f"Gagal: Kolom yang dibutuhkan ({', '.join(required_cols)}) tidak ada di file."

    if "jumlah_kejadian" not in df.columns:
        df["jumlah_kejadian"] = 1
    else:
        df["jumlah_kejadian"] = (
            pd.to_numeric(df["jumlah_kejadian"], errors="coerce").fillna(1).astype(int)
        )

    conn = None
    try:
        conn = sqlite3.connect("database_kejadian.db")
        save_new_event(None, None, None, True)  # Ensure table exists
        df_to_insert = df[
            ["waktu_kejadian", "jenis_kejahatan", "tkp", "jumlah_kejadian"]
        ]
        df_to_insert["waktu_kejadian"] = pd.to_datetime(
            df_to_insert["waktu_kejadian"]
        ).dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_to_insert.to_sql("kejadian", conn, if_exists="append", index=False)

        gr.Success(f"Data {len(df)} berhasil diimpor ke database.")
        return f"Sukses: {len(df)} baris data berhasil diimpor ke database."
    except Exception as e:
        gr.Error(f"Gagal mengimpor data: {e}")
        return f"Gagal mengimpor data: {e}"
    finally:
        if conn:
            conn.close()


def load_data_from_db():
    """Memuat semua data dari tabel 'kejadian' di database SQLite."""
    conn = None
    target_columns = ["waktu_kejadian", "jenis_kejahatan", "tkp", "jumlah_kejadian"]
    try:
        conn = sqlite3.connect("database_kejadian.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kejadian'"
        )
        if cursor.fetchone() is None:
            return pd.DataFrame(
                columns=[
                    "id",
                    "waktu_kejadian",
                    "jenis_kejahatan",
                    "tkp",
                    "jumlah_kejadian",
                ]
            )
        query = f"SELECT {', '.join(target_columns)} FROM kejadian ORDER BY waktu_kejadian DESC"

        return pd.read_sql_query(query, conn)
    except Exception as e:
        return pd.DataFrame({"Error": [f"Gagal memuat data: {e}"]})
    finally:
        if conn:
            conn.close()
