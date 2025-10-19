import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def counts_per_location_type(df, lokasi_col, jenis_col, jumlah_col=None):
    """
    Kembalikan pivot table: index=lokasi, columns=jenis_kejahatan, values=jumlah (atau count).
    - jika jumlah_col diberikan dan numeric -> aggregasi sum
    - jika tidak -> hitung jumlah baris per kombinasi (count)
    """
    if df is None or lokasi_col not in df.columns or jenis_col not in df.columns:
        return pd.DataFrame()  # kosong sebagai fallback

    if jumlah_col and (jumlah_col in df.columns) and pd.api.types.is_numeric_dtype(df[jumlah_col]):
        agg = df.groupby([lokasi_col, jenis_col])[jumlah_col].sum().reset_index()
        pivot = agg.pivot(index=lokasi_col, columns=jenis_col, values=jumlah_col).fillna(0).astype(int)
    else:
        agg = df.groupby([lokasi_col, jenis_col]).size().reset_index(name="count")
        pivot = agg.pivot(index=lokasi_col, columns=jenis_col, values="count").fillna(0).astype(int)

    # urut berdasarkan total kejadian menurun
    pivot["__total__"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__total__", ascending=False)
    pivot = pivot.drop(columns="__total__")
    return pivot

def plot_counts_per_location_type(pivot_df, top_n=12, figsize=(12,6)):
    """
    Gantikan stacked bar dengan heatmap: baris = lokasi (top_n), kolom = jenis kejahatan,
    sel berisi jumlah kejadian. Kembalikan objek matplotlib.figure.
    """
    if pivot_df is None or pivot_df.empty:
        return make_info_fig("Data TKP/jenis tidak memadai untuk membuat grafik.")

    plot_df = pivot_df.head(top_n)
    data = plot_df.values.astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu")

    # ticks/labels
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels([str(c) for c in plot_df.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(r) for r in plot_df.index])

    # anotasi tiap sel
    thresh = data.max() / 2.0 if data.size and data.max() > 0 else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i, j])
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(f"Jumlah Kejahatan per Jenis di Top {min(top_n, len(plot_df))} Lokasi (Heatmap)")
    ax.set_xlabel("Jenis Kejahatan")
    ax.set_ylabel("Lokasi (TKP)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jumlah Kejadian")
    plt.tight_layout()
    return fig


def make_info_fig(text, figsize=(8, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    plt.tight_layout()
    return fig

def plot_top_counts(series, title, xlabel):
    if series is None:
        return make_info_fig(f"Tidak dapat menemukan kolom untuk {title}.")
    series = series.dropna().astype(str).str.strip()
    vc = series.value_counts().head(10)
    if vc.empty:
        return make_info_fig(f"Data kosong untuk {title}.")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    vc.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel("Jumlah Kejadian")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    return fig