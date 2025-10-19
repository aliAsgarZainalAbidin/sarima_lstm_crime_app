from gradio.themes import Soft
import gradio as gr
import pandas as pd
import numpy as np
import sys
from pathlib import Path
# pastikan folder "src" ada di sys.path sehingga import crime_forecast berhasil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import warnings

from crime_forecast.pipeline import run_pipeline
from crime_forecast.gui.gradio_ui import demo

warnings.filterwarnings("ignore")

APP_TITLE = "Hybrid SARIMA–LSTM | Prediksi Kejahatan"
APP_TAGLINE = (
    "Pra-pemrosesan bulanan + pembersihan outlier, EDA (ADF/ACF/PACF), "
    "pelatihan SARIMA + LSTM residual, evaluasi, analisis TKP/Jenis, kalender musiman, dan peta lokasi."
)

def main():
    theme = Soft(primary_hue="blue", secondary_hue="slate").set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_shadow="*shadow_drop_lg",
        block_radius="24px",
        button_large_padding="18px",
        button_large_radius="18px",
        input_radius="14px",
    )

    demo.launch(share=False)

if __name__ == "__main__":
    main()