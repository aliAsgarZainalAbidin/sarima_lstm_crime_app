import pickle
import json
from pathlib import Path
import joblib
import torch
import warnings

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_sarima(result_obj, path):
    """
    result_obj: fitted SARIMAXResults (statsmodels)
    path: file path (str or Path) to write, e.g. models_dir / "sarima.pkl"
    """
    p = ensure_dir(Path(path).parent)
    with open(p / Path(path).name, "wb") as f:
        pickle.dump(result_obj, f)

def load_sarima(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_lstm_state(model, path):
    """
    Save LSTM model state_dict (torch).
    If model is None, skip saving and warn.
    path: file path to write, e.g. models_dir / "lstm_residual.pth"
    """
    if model is None:
        warnings.warn("save_lstm_state: model is None — skipping LSTM save.", UserWarning)
        return None
    p = ensure_dir(Path(path).parent)
    torch.save(model.state_dict(), p / Path(path).name)
    return str(p / Path(path).name)

def load_lstm_state(model_class, path, device="cpu"):
    """
    model_class: callable that returns an instance of the LSTM model (uninitialized)
    path: path to .pth state_dict
    device: "cpu" or "cuda"
    returns: model instance with loaded state_dict on device
    """
    model = model_class()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def save_scaler(scaler_obj, path):
    """
    Save sklearn scaler (joblib)
    """
    p = ensure_dir(Path(path).parent)
    joblib.dump(scaler_obj, p / Path(path).name)

def load_scaler(path):
    return joblib.load(path)

def save_metadata(meta: dict, path):
    p = ensure_dir(Path(path).parent)
    with open(p / Path(path).name, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_hybrid(sarima_obj, lstm_model, scaler=None, models_dir="models/hybrid"):
    """
    Convenience: save SARIMA (pickle), LSTM state_dict (pth), scaler (joblib), and metadata json.
    Returns dict with saved paths.
    """
    md = {}
    d = ensure_dir(models_dir)
    sarima_path = d / "sarima.pkl"
    lstm_path = d / "lstm_residual.pth"
    meta_path  = d / "metadata.json"
    save_sarima(sarima_obj, sarima_path)
    md["sarima"] = str(sarima_path)
    # save LSTM only if provided; save_lstm_state will return None if skipped
    lstm_saved = save_lstm_state(lstm_model, lstm_path)
    if lstm_saved:
        md["lstm"] = lstm_saved
    if scaler is not None:
        scaler_path = d / "scaler.joblib"
        save_scaler(scaler, scaler_path)
        md["scaler"] = str(scaler_path)
    # include minimal metadata
    save_metadata(md, meta_path)
    md["metadata"] = str(meta_path)
    return md