# inference_system/data/data_loader.py

import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

try:
    # Si tienes tu módulo de indicadores dentro del paquete:
    from data.indicators import add_all_indicators
except Exception:
    # Fallback: si lo tienes a nivel raíz
    from indicators import add_all_indicators

# Rutas
DATASET_FOLDER = "datasets"
TRAINING_FOLDER = os.path.join(DATASET_FOLDER, "train")
PREDICTION_FOLDER = os.path.join(DATASET_FOLDER, "predict")

os.makedirs(TRAINING_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas y tipos: Date, Open, High, Low, Close, Volume
    """
    df = df.copy()
    # La descarga multi-ticker trae índice temporal. Lo reseteamos.
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # yfinance pone la columna temporal como 'Date' o 'Datetime' según el intervalo
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "Date"})

    # Columnas esperadas (cuando auto_adjust=True no aparece 'Adj Close')
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    # Tipos
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Algunas ventanas (por ejemplo 1h/15m) pueden no tener Volume -> rellena a 0
    df["Volume"] = df["Volume"].fillna(0)
    # Quita filas totalmente vacías
    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)
    return df

def _download_chunk(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
    interval: str,
    auto_adjust: bool,
    prepost: bool,
    backoff_s: float = 2.0,
    tries: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Descarga un bloque de tickers con reintentos y normaliza por ticker.
    """
    out: Dict[str, pd.DataFrame] = {}
    for attempt in range(1, tries + 1):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            break
        except Exception as e:
            if attempt == tries:
                raise
            time.sleep(backoff_s * attempt)
    # Maneja el caso 1-ticker (df sin MultiIndex) y multi-ticker (MultiIndex en columnas)
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker
        lvl0 = df.columns.get_level_values(0)
        for t in tickers:
            if t not in set(lvl0):
                continue
            sub = df[t].copy()
            sub = _standardize_ohlcv(sub)
            out[t] = sub
    else:
        # 1 ticker o respuesta "aplanada"
        # Intentamos identificar a qué ticker corresponde
        t = tickers[0] if tickers else "TICKER"
        sub = _standardize_ohlcv(df.copy())
        out[t] = sub
    return out

def download_yahoo_bulk(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    prepost: bool = False,
    chunk_size: int = 40,
    tries: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Descarga TODOS los símbolos en BLOQUES (multi-ticker), con reintentos por bloque.
    Devuelve dict {ticker: df_ohlcv}.
    """
    symbols = sorted(list(set(symbols)))
    all_data: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        part = _download_chunk(
            tickers=chunk,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            prepost=prepost,
            tries=tries,
        )
        all_data.update(part)
    return all_data

def save_bulk_to_csv(
    bulk: Dict[str, pd.DataFrame],
    mode: str = "train",
    compute_indicators: bool = True,
) -> Dict[str, str]:
    """
    Guarda un CSV por ticker en la carpeta correspondiente; opcionalmente añade indicadores.
    mode: 'train' | 'predict'
    Retorna mapping {ticker: path_csv}
    """
    folder = TRAINING_FOLDER if mode.lower() == "train" else PREDICTION_FOLDER
    os.makedirs(folder, exist_ok=True)

    out_paths: Dict[str, str] = {}

    for tkr, df in bulk.items():
        df_to_save = df.copy()
        if compute_indicators:
            try:
                df_to_save = add_all_indicators(df_to_save)
            except Exception as e:
                # Si fallan indicadores de un ticker, guardamos al menos OHLCV
                print(f"[WARN] Indicadores fallaron para {tkr}: {e}. Guardo OHLCV.")
                df_to_save = df.copy()

        fname = f"{tkr}_{'train' if mode.lower() == 'train' else 'predict'}.csv"
        path = os.path.join(folder, fname)
        df_to_save.to_csv(path, index=False)
        out_paths[tkr] = path

    return out_paths

def clean_old_files(folder: str, days: int = 60):
    """
    Elimina archivos en 'folder' con fecha de modificación anterior a N días.
    """
    now = time.time()
    cutoff = now - days * 86400  # segundos en N días
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            os.remove(fpath)