import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch

class ResidualLSTM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def make_sequences(arr, win):
    X, y = [], []
    for i in range(win, len(arr)):
        X.append(arr[i - win : i])
        y.append(arr[i])
    return np.array(X, np.float32), np.array(y, np.float32)

def fit_lstm_and_predict(residual_series, horizon, window=6, hidden=124, lr=0.001,
                         batch_size=32, epochs=200, patience=13):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    res_scaled = scaler.fit_transform(residual_series.values.reshape(-1, 1)).astype(np.float32).flatten()
    X_np, y_np = make_sequences(res_scaled, window)
    if len(X_np) < 5:
        raise ValueError("Residual terlalu pendek untuk LSTM. Kurangi window atau tambah data.")
    split = max(5, int(0.8 * len(X_np)))
    Xtr = torch.from_numpy(X_np[:split]).unsqueeze(-1)
    ytr = torch.from_numpy(y_np[:split]).unsqueeze(-1)
    Xval = torch.from_numpy(X_np[split:]).unsqueeze(-1)
    yval = torch.from_numpy(y_np[split:]).unsqueeze(-1)

    model = ResidualLSTM(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    best_loss, wait, best_state = float("inf"), 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xval), yval).item() if len(Xval) > 0 else float("inf")
        if vloss < best_loss - 1e-8:
            best_loss, wait = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    preds_scaled, w = [], res_scaled[-window:].tolist()
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(w, dtype=torch.float32).view(1, window, 1)
            yhat = model(x).item()
            preds_scaled.append(yhat)
            w = w[1:] + [yhat]
    pred_res = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return pred_res