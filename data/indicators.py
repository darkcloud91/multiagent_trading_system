# indicators.py

import pandas as pd
import ta  # Technical Analysis library

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df_ind = df.copy()
    
    # Asegúrate que los nombres de columnas son los esperados
    df_ind = ta.add_all_ta_features(
        df_ind,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )
    return df_ind