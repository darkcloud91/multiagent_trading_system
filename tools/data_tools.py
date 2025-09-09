from typing import Dict, Any, List
import os
import pandas as pd
from typing import Dict, List, Optional
import os

from pydantic import BaseModel

#--- Dependencias de los módulos del proyecto ---
from data.nasdaq100 import NASDAQ_100_TICKERS
from data.indicators import add_all_indicators

# --- Lógica de Fallback (movida desde data_agent.py) ---
# Intenta usar el método de descarga masiva; si no, prepara el método individual.
try:
    from data.data_loader import download_yahoo_bulk, save_bulk_to_csv
    HAVE_BULK = True
except ImportError:
    HAVE_BULK = False
    import yfinance as yf
    import pandas as pd

    def _download_one(symbol: str, **kwargs) -> pd.DataFrame:
        df = yf.download(symbol, progress=False, **kwargs)
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[cols].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        return df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

# --- Funciones de Ayuda (movidas desde data_agent.py) ---

def _resolve_symbols(raw: Optional[List[str]]) -> List[str]:
    """Admite None, [], 'NASDAQ100', o una lista normal de tickers."""
    if not raw or "NASDAQ100" in [str(s).upper() for s in raw]:
        return NASDAQ_100_TICKERS[:]
    
    seen, out = set(), []
    for t in raw:
        t_upper = str(t).upper().strip()
        if t_upper and t_upper not in seen:
            seen.add(t_upper)
            out.append(t_upper)
    return out

# --- Herramientas que el Agente podrá usar ---

def fetch_and_process_market_data(
    symbols: Optional[List[str]] = None,
    mode: str = "train",
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    compute_indicators: bool = True
) -> Dict[str, Any]:
    """
    Downloads, processes, and saves historical stock market data for one or more symbols.
    It uses an efficient bulk download method if available, otherwise, it falls back
    to downloading symbols one by one.

    Args:
        symbols: List of stock tickers. If None or includes "NASDAQ100", defaults to the full NASDAQ-100 list.
        mode: "train" or "predict". Determines the subfolder for saving the data.
        period: Time period to download (e.g., "5y", "1mo", "max"). Overrides start/end dates.
        start_date: Start date for the data in "YYYY-MM-DD" format.
        end_date: End date for the data in "YYYY-MM-DD" format.
        interval: Data interval (e.g., "1d", "1h", "1wk").
        compute_indicators: If True, calculates and adds a comprehensive set of technical indicators.
    """
    resolved_symbols = _resolve_symbols(symbols)
    
    # Determina el rango temporal
    if not period and not start_date:
        period = "10y"

    if HAVE_BULK:
        print("Usando el método de descarga masiva (BULK)...")
        bulk_data = download_yahoo_bulk(
            symbols=resolved_symbols, start=start_date, end=end_date, period=period,
            interval=interval, auto_adjust=True, prepost=False,
        )
        saved_paths = save_bulk_to_csv(
            bulk=bulk_data, mode=mode, compute_indicators=compute_indicators
        )
        # Construir el diccionario de resultados para el modo bulk
        results: Dict[str, Dict] = {}
        for sym in resolved_symbols:
            if sym not in bulk_data:
                results[sym] = {"status": "missing", "message": "Sin datos de Yahoo."}
                continue
            df = bulk_data[sym]
            results[sym] = {
                "status": "ok", "rows": len(df), "path": saved_paths.get(sym)
            }
        return {"summary": "Proceso BULK completado.", "details": results}

    # Lógica de Fallback (si HAVE_BULK es False)
    print("Usando el método de descarga individual (FALLBACK)...")
    folder = os.path.join("datasets", mode)
    os.makedirs(folder, exist_ok=True)
    results_fallback: Dict[str, Dict] = {}
    for sym in resolved_symbols:
        try:
            kwargs = dict(interval=interval, auto_adjust=True)
            if period:
                kwargs["period"] = period
            else:
                kwargs["start"] = start_date
                kwargs["end"] = end_date
            df = _download_one(sym, **kwargs)
            if compute_indicators:
                df = add_all_indicators(df)
            
            path = os.path.join(folder, f"{sym}_{mode}.csv")
            df.to_csv(path, index=False)
            results_fallback[sym] = {"status": "ok", "rows": len(df), "path": path}
        except Exception as e:
            results_fallback[sym] = {"status": "error", "message": str(e)}
            
    return {"summary": "Proceso FALLBACK completado.", "details": results_fallback}

def list_available_datasets() -> Dict[str, Any]:
    """
    Scans the local filesystem and returns a list of all available CSV datasets
    in the 'datasets' directory.
    """
    folder = os.path.join("datasets")
    if not os.path.exists(folder):
        return {"status": "success", "message": "La carpeta de datasets no existe.", "datasets": []}
    
    found_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                found_files.append(os.path.join(root, f))
    
    return {
        "status": "success",
        "message": f"Se encontraron {len(found_files)} datasets.",
        "datasets": found_files
    }