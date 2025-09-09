import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional
from sklearn.metrics import f1_score

from data.nasdaq100 import NASDAQ_100_TICKERS
from ml.transformer import TimeSeriesTransformer

# ---------------- Config ----------------
UP_THR = 0.03        # > +3%  -> clase 2
DOWN_THR = -0.01     # < -1%  -> clase 0
N_FUTURE = 5
SEQ_LEN = 30
TRAIN_FRAC = 0.8

DATA_DIR = os.path.join("datasets", "train")
MODELS_DIR = os.path.join("models")
os.makedirs(MODELS_DIR, exist_ok=True)
# ----------------------------------------

def _assign_3class_labels(future_ret: pd.Series, down_thr=-0.01, up_thr=0.03):
    labels = np.full(len(future_ret), 1, dtype=np.int64)
    labels[future_ret < down_thr] = 0
    labels[future_ret > up_thr] = 2
    return labels

def _load_all_train_csvs(symbols):
    dfs = []
    for sym in symbols:
        path = os.path.join(DATA_DIR, f"{sym}_train.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Falta {path}")
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise ValueError(f"{sym}: falta 'Date'")
        df["Date"] = pd.to_datetime(df["Date"])
        df["Ticker"] = sym
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full.sort_values(["Date", "Ticker"], inplace=True)
    return full

def _build_multiasset_sequences(full_df: pd.DataFrame,
                                n_future=N_FUTURE, seq_len=SEQ_LEN, train_frac=TRAIN_FRAC):
    df = full_df.copy()
    df["Future_Return"] = df.groupby("Ticker")["Close"].shift(-n_future) / df["Close"] - 1.0
    df = df.dropna(subset=["Future_Return"]).copy()
    df["Label"] = _assign_3class_labels(df["Future_Return"], DOWN_THR, UP_THR)

    unique_dates = np.sort(df["Date"].unique())
    split_idx = int(len(unique_dates) * float(train_frac))
    split_idx = max(split_idx, seq_len + 1)
    split_date = unique_dates[split_idx - 1]

    ignore_cols = {"Future_Return", "Label", "Ticker"}
    feature_cols = [c for c in df.columns
                    if c not in ignore_cols and c != "Date" and np.issubdtype(df[c].dtype, np.number)]

    scaler = StandardScaler()
    train_mask = df["Date"] <= split_date
    df.loc[train_mask, feature_cols] = scaler.fit_transform(df.loc[train_mask, feature_cols])
    df.loc[~train_mask, feature_cols] = scaler.transform(df.loc[~train_mask, feature_cols])

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    tickers = sorted(df["Ticker"].unique().tolist())
    ticker2id = {t: i for i, t in enumerate(tickers)}
    with open(os.path.join(MODELS_DIR, "ticker2id.json"), "w") as f:
        json.dump(ticker2id, f, indent=2)

    with open(os.path.join(MODELS_DIR, "labeling_config.json"), "w") as f:
        json.dump({
            "method": "fixed_thresholds",
            "n_future": n_future,
            "seq_len": seq_len,
            "train_frac": train_frac,
            "down_thr": DOWN_THR,
            "up_thr": UP_THR,
            "class_names": {0: "DOWN<-1%", 1: "FLAT[-1%,+3%]", 2: "UP>+3%"},
            "split_date": str(pd.to_datetime(split_date).date())
        }, f, indent=2)

    X_tr, A_tr, y_tr = [], [], []
    X_va, A_va, y_va = [], [], []
    for tkr, g in df.groupby("Ticker", sort=False):
        g = g.sort_values("Date")
        aid = ticker2id[tkr]
        for i in range(len(g) - seq_len):
            label_idx = i + seq_len
            label_date = g["Date"].iloc[label_idx]
            x_seq = g[feature_cols].iloc[i:i+seq_len].values.astype(np.float32)
            y_lbl = int(g["Label"].iloc[label_idx])
            if label_date <= split_date:
                X_tr.append(x_seq); A_tr.append(aid); y_tr.append(y_lbl)
            else:
                X_va.append(x_seq); A_va.append(aid); y_va.append(y_lbl)

    X_tr = np.asarray(X_tr, dtype=np.float32)
    A_tr = np.asarray(A_tr, dtype=np.int64)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    X_va = np.asarray(X_va, dtype=np.float32)
    A_va = np.asarray(A_va, dtype=np.int64)
    y_va = np.asarray(y_va, dtype=np.int64)

    return (X_tr, A_tr, y_tr), (X_va, A_va, y_va), len(tickers), feature_cols

def compute_sample_weights(y, n_classes=3):
    
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w_per_class = (counts.sum() / (counts + 1e-6))
    w_per_class = w_per_class / (w_per_class.mean() + 1e-12)
    return w_per_class, (1.0 / (counts[y] + 1e-6))

def train_multiasset(symbols=None,
                     model_path=os.path.join(MODELS_DIR, "nasdaq100_transformer.pt"),
                     max_epochs=80, patience=8, batch_size=128, base_lr=3e-4, weight_decay=1e-2):
    
    if not symbols:
        symbols = NASDAQ_100_TICKERS[:]

    full = _load_all_train_csvs(symbols)
    (X_tr, A_tr, y_tr), (X_va, A_va, y_va), num_assets, feature_cols = _build_multiasset_sequences(full)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        input_size=len(feature_cols), num_classes=3,
        d_model=128, nhead=8, num_layers=4, dim_ff=512, dropout=0.2,
        num_assets=num_assets,
    ).to(device)

    w_class, w_sample = compute_sample_weights(y_tr, n_classes=3)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(w_class, dtype=torch.float32, device=device),
        label_smoothing=0.05
    )

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(w_sample / (w_sample.mean() + 1e-12), dtype=torch.double),
        num_samples=len(w_sample),
        replacement=True
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(A_tr, dtype=torch.long),
            torch.tensor(y_tr, dtype=torch.long),
        ),
        batch_size=batch_size, sampler=sampler, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_va, dtype=torch.float32),
            torch.tensor(A_va, dtype=torch.long),
            torch.tensor(y_va, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=base_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs,
        pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
    )

    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, ab, yb in train_loader:
            xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6).to(device)
            ab = ab.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, asset_ids=ab)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            run_loss += loss.item() * yb.size(0)
        train_loss = run_loss / len(train_loader.dataset)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, ab, yb in val_loader:
                xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6).to(device)
                ab = ab.to(device)
                yb = yb.to(device)
                logits = model(xb, asset_ids=ab)
                preds = torch.argmax(logits, dim=1)
                y_true.append(yb.cpu().numpy())
                y_pred.append(preds.cpu().numpy())
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"Epoch {epoch:02d} | TrainLoss {train_loss:.4f} | F1-macro {f1_macro:.4f}")

        if f1_macro > best_f1 + 1e-6:
            best_f1, best_state = f1_macro, model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"⏹️ Early stopping (best F1={best_f1:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    print("✅ Modelo guardado en:", model_path)

def train_model_handler_multi(symbols: Optional[List[str]] = None):
    """
    ✅ NUEVO: Docstring para el agente
    Initiates the complete training pipeline for the multi-asset Transformer model.

    This is a long-running, computationally intensive process that will overwrite any
    previously trained models. It loads all necessary training data, preprocesses it,
    trains the model with early stopping, and saves the final model and all necessary
    artifacts (scaler, configs) to the 'models' directory.

    Args:
        symbols: An optional list of stock tickers to use for training. If not provided,
                 it defaults to the full NASDAQ-100 list.
    """
    try:
        train_multiasset(symbols)
        return {"status": "success", "message": "Entrenamiento multi-activo completado con éxito."}
    except Exception as e:
        return {"status": "error", "message": f"El entrenamiento falló: {e}"}
    
    
def check_existing_model():
    """
    Checks if a trained model and its associated artifacts already exist
    in the 'trading_system/models' directory.

    Returns a summary of which files were found, including the main model file
    and other necessary artifacts like the scaler.
    """
    model_path = os.path.join(MODELS_DIR, "nasdaq100_transformer.pt")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_cols.json")
    
    found_files = {
        "model_exists": os.path.exists(model_path),
        "scaler_exists": os.path.exists(scaler_path),
        "features_exist": os.path.exists(features_path)
    }
    
    if all(found_files.values()):
        message = "Se ha encontrado un modelo entrenado y todos sus artefactos necesarios."
    elif any(found_files.values()):
        message = "Se han encontrado algunos artefactos, pero el modelo podría estar incompleto."
    else:
        message = "No se ha encontrado ningún modelo entrenado."

    return {"status": "success", "message": message, "files_check": found_files}
