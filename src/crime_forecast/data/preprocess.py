import pandas as pd


def infer_columns(df: pd.DataFrame):
    date_candidates = [
        c
        for c in df.columns
        if str(c).strip().lower()
        in [
            "tanggal",
            "tgl",
            "date",
            "time",
            "waktu",
            "bulan",
            "month",
            "period",
            "periode",
            "waktu_kejadian",
        ]
    ]
    value_candidates = [
        c
        for c in df.columns
        if str(c).strip().lower()
        in [
            "jumlah",
            "nilai",
            "value",
            "count",
            "kejahatan",
            "kasus",
            "cases",
            "y",
            "total",
            "jumlah_kejadian",
        ]
    ]
    date_col = date_candidates[0] if date_candidates else None
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    value_col = None
    for c in value_candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            value_col = c
            break
    if value_col is None and numeric_cols:
        value_col = numeric_cols[0]
    return date_col, value_col


def ensure_monthly_sum(df, date_col, value_col):
    """
    Pastikan dataframe memiliki indeks bulanan (MS) dan kolom 'value'.
    Jika date_col/value_col tidak ada, coba infer berdasarkan nama atau tipe kolom.
    """
    if df is None or len(df.columns) == 0:
        raise ValueError("Data kosong. Pastikan file/table berisi data.")

    # coba gunakan nama yang diberikan jika valid
    if (date_col in df.columns) and (value_col in df.columns):
        sel_date, sel_value = date_col, value_col
    else:
        # daftar kandidat nama umum (lowercase, strip)
        col_map = {c: str(c).strip().lower() for c in df.columns}
        date_candidates = [
            k
            for k, v in col_map.items()
            if v
            in [
                "tanggal",
                "tgl",
                "date",
                "time",
                "waktu",
                "bulan",
                "month",
                "period",
                "periode",
                "waktu_kejadian",
            ]
        ]
        value_candidates = [
            k
            for k, v in col_map.items()
            if v
            in [
                "jumlah",
                "nilai",
                "value",
                "count",
                "kejahatan",
                "kasus",
                "cases",
                "y",
                "total",
                "jumlah_kejadian",
            ]
        ]

        sel_date = date_candidates[0] if date_candidates else None
        sel_value = value_candidates[0] if value_candidates else None

        # fallback berdasarkan tipe kolom
        if sel_date is None:
            sel_date = next(
                (
                    c
                    for c in df.columns
                    if pd.api.types.is_datetime64_any_dtype(df[c])
                    or pd.api.types.is_object_dtype(df[c])
                    and pd.to_datetime(df[c], errors="coerce").notna().any()
                ),
                None,
            )
        if sel_value is None:
            sel_value = next(
                (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None
            )

    if sel_date is None or sel_value is None:
        raise ValueError(
            "Tidak dapat menemukan kolom tanggal/nilai. Periksa nama kolom atau isi parameter kolom secara eksplisit."
        )

    # pilih dan normalisasi
    out = (
        df[[sel_date, sel_value]]
        .rename(columns={sel_date: "date", sel_value: "value"})
        .copy()
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = (
        out.sort_values("date")
        .set_index("date")
        .resample("MS")
        .sum()
        .asfreq("MS")
        .fillna(0)
    )
    return out


def remove_outliers_iqr(ts: pd.DataFrame):
    q1, q3 = ts["value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ts[(ts["value"] >= low) & (ts["value"] <= high)]
