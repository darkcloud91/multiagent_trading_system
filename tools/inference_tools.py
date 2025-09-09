# tools/inference_tools.py

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import datetime
from typing import Dict, Any, List, Optional

from google.adk.tools.tool_context import ToolContext
from data.nasdaq100 import NASDAQ_100_TICKERS
from ml.transformer import TimeSeriesTransformer

# --- Rutas y Configuración ---
MODELS_DIR = os.path.join("models")
PRED_DIR = os.path.join("datasets", "predict")
OUTPUTS_DIR = os.path.join("outputs")

# --- Funciones Auxiliares ---
def _load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "r") as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODELS_DIR, "ticker2id.json"), "r") as f:
        ticker2id = json.load(f)
    with open(os.path.join(MODELS_DIR, "labeling_config.json"), "r") as f:
        cfg = json.load(f)
    return scaler, feature_cols, ticker2id, cfg

def _resolve_symbols(symbols):
    # ... (el código de esta función no cambia)
    if symbols is None or (isinstance(symbols, list) and len(symbols) == 0):
        return NASDAQ_100_TICKERS[:]
    if isinstance(symbols, str) and symbols.upper() == "NASDAQ100":
        return NASDAQ_100_TICKERS[:]
    if isinstance(symbols, list) and len(symbols) == 1 and str(symbols[0]).upper() == "NASDAQ100":
        return NASDAQ_100_TICKERS[:]
    return symbols

def save_inference_output(output, folder=OUTPUTS_DIR):
    os.makedirs(folder, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(folder, f"inference_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path

# --- Herramienta Principal para el Agente ---

def infer_multi(
    tool_context: ToolContext,
    symbols: Optional[List[str]] = None,
    sequence_length: int = 30
) -> Dict[str, Any]:
    """
    Runs inference using the trained Transformer model to predict future price movements
    for a list of stock symbols.

    This tool loads the trained model and necessary artifacts, processes the latest
    prediction data for each symbol, and generates probabilities for UP, DOWN, and FLAT movements.
    It then ranks the symbols based on their probability of going UP and saves the full
    output to a timestamped JSON file.

    Args:
        tool_context: The context object provided by the ADK agent.
        symbols: An optional list of stock tickers to predict. If not provided or if "NASDAQ100"
                 is included, it defaults to the full NASDAQ-100 list.
        sequence_length: The number of recent data points (days) to use for the prediction sequence.
                         This must match the sequence length used during training.
    """
    model_path = os.path.join(MODELS_DIR, "nasdaq100_transformer.pt")
    resolved_symbols = _resolve_symbols(symbols)
    scaler, feature_cols, ticker2id, cfg = _load_artifacts()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        input_size=len(feature_cols), num_classes=3,
        d_model=128, nhead=8, num_layers=4, dim_ff=512, dropout=0.2,
        num_assets=len(ticker2id),
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []
    # ... (el resto de la lógica de la función se mantiene exactamente igual)
    for sym in resolved_symbols:
        csv_path = os.path.join(PRED_DIR, f"{sym}_predict.csv")
        if not os.path.exists(csv_path):
            results.append({"symbol": sym, "error": f"No existe {csv_path}"})
            continue
        if sym not in ticker2id:
            results.append({"symbol": sym, "error": f"{sym} no está en ticker2id (¿faltó en train?)"})
            continue

        df = pd.read_csv(csv_path)
        if len(df) < sequence_length:
            results.append({"symbol": sym, "error": f"Filas insuficientes ({len(df)}) para seq={sequence_length}"})
            continue

        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])

        last_seq = df_scaled[feature_cols].iloc[-sequence_length:].values.astype(np.float32)
        X = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        A = torch.tensor([ticker2id[sym]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(X, asset_ids=A)
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

        pred = int(np.argmax(probs))
        cls_map = cfg.get("class_names", {0: "DOWN", 1: "FLAT", 2: "UP"})
        results.append({
            "symbol": sym,
            "class": cls_map[str(pred)],
            "p_down": float(probs[0]),
            "p_flat": float(probs[1]),
            "p_up": float(probs[2]),
            "score_up_minus_down": float(probs[2] - probs[0])
        })
    
    ok = [r for r in results if "p_up" in r]
    ranked = sorted(ok, key=lambda r: r["score_up_minus_down"], reverse=True)
    output = {
        "ranking_up": ranked,
        "all_results": results,
        "thresholds": {"down_thr": cfg.get("down_thr", -0.01), "up_thr": cfg.get("up_thr", 0.03)}
    }

    saved_path = save_inference_output(output)
    output["saved_json"] = saved_path

    return output