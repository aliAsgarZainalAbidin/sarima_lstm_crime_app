import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import json
import numpy as np


def read_coords_file(file_path_or_obj):
    if file_path_or_obj is None:
        return None
    try:
        # Handle both file path (str) and Gradio file object
        if isinstance(file_path_or_obj, str):
            filepath = file_path_or_obj
        else:
            filepath = file_path_or_obj.name

        name = filepath.lower()
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        cols = {c: str(c).strip().lower() for c in df.columns}
        tkp = next(
            (
                k
                for k, v in cols.items()
                if v in ["tkp", "lokasi", "tempat", "location", "nama"]
            ),
            None,
        )
        lat = next((k for k, v in cols.items() if v in ["lat", "latitude", "y"]), None)
        lon = next(
            (
                k
                for k, v in cols.items()
                if v in ["lon", "lng", "long", "longitude", "x"]
            ),
            None,
        )
        if tkp and lat and lon:
            out = df[[tkp, lat, lon]].dropna()
            out.columns = ["tkp", "lat", "lon"]
            out["tkp_norm"] = out["tkp"].astype(str).str.strip().str.lower()
            return out
    except Exception:
        return None
    return None


def make_map_html(
    data_event,
    tkp_col,
    coords_df=None,
    geojson_path=None,
    jenis_col=None,
    jumlah_col=None,
):
    if tkp_col is None or tkp_col not in data_event.columns:
        return "<div style='padding:12px'>Kolom TKP/Lokasi tidak ditemukan.</div>"

    counts = (
        data_event[tkp_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .reset_index()
    )
    counts.columns = ["tkp_norm", "count"]

    # prepare breakdown per jenis if available
    breakdown_map = {}
    if jenis_col and (jenis_col in data_event.columns):
        de = data_event.copy()
        de["tkp_norm"] = de[tkp_col].astype(str).str.strip().str.lower()
        if (
            jumlah_col
            and (jumlah_col in de.columns)
            and pd.api.types.is_numeric_dtype(de[jumlah_col])
        ):
            grp = de.groupby(["tkp_norm", jenis_col])[jumlah_col].sum().reset_index()
            for _, row in grp.iterrows():
                breakdown_map.setdefault(row["tkp_norm"], {})[str(row[jenis_col])] = (
                    int(row[jumlah_col])
                )
        else:
            grp = de.groupby(["tkp_norm", jenis_col]).size().reset_index(name="count")
            for _, row in grp.iterrows():
                breakdown_map.setdefault(row["tkp_norm"], {})[str(row[jenis_col])] = (
                    int(row["count"])
                )

    if coords_df is not None and not coords_df.empty:
        mdf = counts.merge(
            coords_df[["tkp_norm", "lat", "lon"]], on="tkp_norm", how="inner"
        )
        if mdf.empty:
            return "<div style='padding:12px'>Koordinat tidak cocok dengan nama TKP. Cek ejaan di file koordinat.</div>"

        center = [mdf["lat"].median(), mdf["lon"].median()]
        m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

        mc = MarkerCluster().add_to(m)
        for _, r in mdf.iterrows():
            rad = 5 + 2 * np.sqrt(max(r["count"], 1))

            # build popup HTML with breakdown per-type (if available)
            popup_lines = [
                f"<div style='font-weight:600'>{r['tkp_norm'].title()}</div>",
                f"<div style='font-size:12px; color:#555'>Total: {int(r['count'])} kejadian</div>",
            ]
            brk = breakdown_map.get(r["tkp_norm"], {})
            if brk:
                popup_lines.append("<ul style='margin:6px 0 0 18px; padding:0;'>")
                for k, v in sorted(brk.items(), key=lambda x: -int(x[1])):
                    popup_lines.append(
                        f"<li style='font-size:12px; margin:2px 0'>{k}: {v}</li>"
                    )
                popup_lines.append("</ul>")
            popup_html = "".join(popup_lines)

            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=float(rad),
                popup=folium.Popup(popup_html, max_width=300),
                fill=True,
                color="#2b6cb0",
                fill_opacity=0.6,
            ).add_to(mc)

        HeatMap(
            mdf[["lat", "lon", "count"]].values.tolist(), radius=18, blur=15
        ).add_to(m)

        if geojson_path:
            try:
                with open(geojson_path, "r", encoding="utf-8") as f:
                    gj = json.load(f)
                candidate_keys = [
                    "name",
                    "nama",
                    "namobj",
                    "kecamatan",
                    "district",
                    "kec",
                    "kelurahan",
                ]
                rows = []
                for feat in gj.get("features", []):
                    props = feat.get("properties", {})
                    key = next(
                        (
                            k
                            for k in candidate_keys
                            if k in {kk.lower(): vv for kk, vv in props.items()}
                        ),
                        None,
                    )
                    if key is None:
                        lprops = {kk.lower(): vv for kk, vv in props.items()}
                        key = next((k for k in candidate_keys if k in lprops), None)
                        val = lprops.get(key) if key else None
                    else:
                        val = {kk.lower(): vv for kk, vv in props.items()}.get(key)
                    if val is None:
                        continue
                    rows.append({"wilayah": str(val).strip().lower()})
                if rows:
                    df_geo = pd.DataFrame(rows).drop_duplicates()
                    agg = []
                    for w in df_geo["wilayah"]:
                        cnt = counts[counts["tkp_norm"].str.contains(w, regex=False)][
                            "count"
                        ].sum()
                        agg.append({"wilayah": w, "count": int(cnt)})
                    df_ch = pd.DataFrame(agg)
                    if df_ch["count"].sum() > 0:
                        folium.Choropleth(
                            geo_data=gj,
                            data=df_ch,
                            columns=["wilayah", "count"],
                            key_on="feature.properties."
                            + (
                                "name"
                                if "name" in gj["features"][0]["properties"]
                                else next(
                                    (k for k in gj["features"][0]["properties"].keys()),
                                    "name",
                                )
                            ),
                            fill_color="YlOrRd",
                            fill_opacity=0.6,
                            line_opacity=0.5,
                            legend_name="Perkiraan Kepadatan Kejadian (match nama wilayah)",
                        ).add_to(m)
            except Exception:
                pass

        return m._repr_html_()

    return "<div style='padding:12px'>Unggah file koordinat (CSV/XLSX) dengan kolom: <b>tkp, lat, lon</b> untuk menampilkan peta.</div>"


def make_map_html_proportion(
    df_raw,             # Data mentah historis
    future_dates,
    future_pred,
    tkp_col,
    jumlah_col,
    coords_df=None,
    coords_file=None,
    geojson_path=None,
    target_bulan_str=None,
):
    counts = (
        df_raw[tkp_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .reset_index()
    )
    counts.columns = ["tkp_norm", "count"]

    df_future = pd.DataFrame({
        "bulan": future_dates,
        "prediksi": future_pred
    })
    df_future["bulan_str"] = df_future["bulan"].dt.strftime("%B %Y")

    if target_bulan_str not in df_future["bulan_str"].values:
        print(f"Warning: '{target_bulan_str}' tidak ditemukan di df_future. Dipakai bulan terakhir horizon.")
        row_target = df_future.iloc[-1]
        target_bulan_str = row_target["bulan_str"]
    else:
        row_target = df_future[df_future["bulan_str"] == target_bulan_str].iloc[0]

    pred_total = float(row_target["prediksi"])

    if tkp_col not in df_raw.columns or jumlah_col not in df_raw.columns:
        return "<div style='padding:12px'>Kolom TKP atau Jumlah tidak ditemukan di data historis.</div>"

    df_hist = df_raw.copy()
    df_hist[jumlah_col] = pd.to_numeric(df_hist[jumlah_col], errors="coerce").fillna(0)

    # total 5 tahun per TKP
    prop = (
        df_hist
        .groupby(tkp_col)[jumlah_col]
        .sum()
        .sort_values(ascending=False)
    )
    total_hist = prop.sum()
    if total_hist == 0:
        raise ValueError("Total historis = 0, cek data.")

    prop = prop / total_hist  # jadi proporsi 0–1
    prop_df = prop.reset_index().rename(columns={"jumlah_kejadian": "proporsi"})
    prop_df["tkp"] = prop_df["tkp"].str.strip().str.lower()


    df_pred_kec = prop_df.copy()
    df_pred_kec["prediksi_kasus"] = df_pred_kec["proporsi"] * pred_total

    df_loc = pd.read_csv(coords_file).copy()

    df_loc.columns = df_loc.columns.str.strip().str.lower()
    if not {"lokasi", "lat", "lon"}.issubset(df_loc.columns):
        raise ValueError("File koordinat harus punya kolom: 'lokasi', 'lat', 'lon'.")

    df_loc["lokasi"] = df_loc["lokasi"].str.strip().str.lower()

    df_map = pd.merge(
        df_loc,
        df_pred_kec,
        left_on="lokasi",
        right_on="tkp",
        how="left"
    )

    df_map["prediksi_kasus"] = df_map["prediksi_kasus"].fillna(0.0)
    df_map["proporsi"] = df_map["proporsi"].fillna(0.0)


    # -------------------------------------------------
    # 4) BUAT PETA FOLIUM (SEMUA KECAMATAN, 1 BULAN HORIZON)
    # -------------------------------------------------
    mdf = counts.merge(
            coords_df[["tkp_norm", "lat", "lon"]], on="tkp_norm", how="inner"
        )
    if mdf.empty:
        return "<div style='padding:12px'>Koordinat tidak cocok dengan nama TKP. Cek ejaan di file koordinat.</div>"

    center = [mdf["lat"].median(), mdf["lon"].median()]

    m = folium.Map(location=center, zoom_start=11)
    cluster = MarkerCluster().add_to(m)

    for _, row in df_map.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        popup_html = (
            f"<b>{row['lokasi'].upper()}</b><br>"
            f"Bulan: {target_bulan_str}<br>"
            f"Prediksi Kasus: {row['prediksi_kasus']:.0f}<br>"
            f"Persentase Kejahatan Per Wilayah: {row['proporsi']*100:.0f}%"
        )   

        popup_obj = folium.Popup(popup_html, max_width=300)

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=popup_obj,
            tooltip=f"{row['lokasi'].title()} ({row['prediksi_kasus']:.0f} kasus)"
        ).add_to(cluster)

    heat_data = mdf[["lat", "lon", "count"]].values.tolist()
    HeatMap(heat_data, radius=20, blur=15).add_to(m)

    # Choropleth Logic (Sama seperti sebelumnya, disederhanakan)
    if geojson_path:
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            candidate_keys = [
                "name",
                "nama",
                "namobj",
                "kecamatan",
                "district",
                "kec",
                "kelurahan",
            ]
            rows = []
            for feat in gj.get("features", []):
                props = feat.get("properties", {})
                key = next(
                    (
                        k
                        for k in candidate_keys
                        if k in {kk.lower(): vv for kk, vv in props.items()}
                    ),
                    None,
                )
                if key is None:
                    lprops = {kk.lower(): vv for kk, vv in props.items()}
                    key = next((k for k in candidate_keys if k in lprops), None)
                    val = lprops.get(key) if key else None
                else:
                    val = {kk.lower(): vv for kk, vv in props.items()}.get(key)
                if val is None:
                    continue
                rows.append({"wilayah": str(val).strip().lower()})
            if rows:
                df_geo = pd.DataFrame(rows).drop_duplicates()
                agg = []
                for w in df_geo["wilayah"]:
                    cnt = counts[counts["tkp_norm"].str.contains(w, regex=False)][
                        "count"
                    ].sum()
                    agg.append({"wilayah": w, "count": int(cnt)})
                df_ch = pd.DataFrame(agg)
                if df_ch["count"].sum() > 0:
                    folium.Choropleth(
                        geo_data=gj,
                        data=df_ch,
                        columns=["wilayah", "count"],
                        key_on="feature.properties."
                        + (
                            "name"
                            if "name" in gj["features"][0]["properties"]
                            else next(
                                (k for k in gj["features"][0]["properties"].keys()),
                                "name",
                            )
                        ),
                        fill_color="YlOrRd",
                        fill_opacity=0.6,
                        line_opacity=0.5,
                        legend_name="Perkiraan Kepadatan Kejadian (match nama wilayah)",
                    ).add_to(m)
        except Exception:
            pass

    return m._repr_html_()
    return "<div style='padding:12px'>Unggah file koordinat (CSV/XLSX) dengan kolom: <b>tkp, lat, lon</b> untuk menampilkan peta.</div>"