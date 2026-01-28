# src/utils/preprocessing.py

import numpy as np
import pandas as pd


def alinear_e_interpolar_series(
    series_dict,
    freq="B",
    metodo_relleno="ffill",
    nombre_indice_fecha="fecha",
    verbose=True
):
    """
    Align and clean multiple time series on a common time index,
    applying forward-fill interpolation without future leakage.

    Args:
        series_dict (dict):
            Dictionary of time series with structure:
            {
                'name': {
                    'df': pd.DataFrame,
                    'fecha_col': str (optional),
                    'valor_col': str (required)
                }
            }
        freq (str): Target date frequency (e.g. 'B', 'D', 'M').
        metodo_relleno (str or None): Gap-filling method. Only 'ffill' is supported.
        nombre_indice_fecha (str): Name of the datetime index.
        verbose (bool): If True, prints alignment summary.

    Returns:
        pd.DataFrame: Aligned and cleaned DataFrame with one column per series.
    """
    series_limpias = {}
    min_fecha, max_fecha = None, None

    # --- Clean individual series ---
    for nombre, cfg in series_dict.items():
        df = cfg["df"].copy()
        valor_col = cfg["valor_col"]
        fecha_col = cfg.get("fecha_col", None)

        if fecha_col is None:
            fecha_col = inferir_columna_fecha(df)

        df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")
        df = df.dropna(subset=[fecha_col]).sort_values(fecha_col)

        valores = (
            df[valor_col]
            .replace(["N/E", "n/e", "NE", "ne", "N.A.", "NA"], np.nan)
        )
        valores = pd.to_numeric(valores, errors="coerce")

        df_limpio = pd.DataFrame({
            nombre_indice_fecha: df[fecha_col],
            nombre: valores
        }).set_index(nombre_indice_fecha).sort_index()

        min_fecha = df_limpio.index.min() if min_fecha is None else min(min_fecha, df_limpio.index.min())
        max_fecha = df_limpio.index.max() if max_fecha is None else max(max_fecha, df_limpio.index.max())

        series_limpias[nombre] = df_limpio

    if min_fecha is None or max_fecha is None:
        raise ValueError("Could not determine common date range.")

    # --- Build common index ---
    if freq is None:
        ref = next(iter(series_limpias.values()))
        freq = pd.infer_freq(ref.index) or "D"

    idx_comun = pd.date_range(start=min_fecha, end=max_fecha, freq=freq)

    # --- Reindex and forward fill ---
    df_final = pd.DataFrame(index=idx_comun)
    df_final.index.name = nombre_indice_fecha

    for nombre, df_limpio in series_limpias.items():
        serie = df_limpio.reindex(idx_comun)[nombre]
        if metodo_relleno == "ffill":
            serie = serie.ffill()
        df_final[nombre] = serie

    if verbose:
        print(f"Common range: {df_final.index.min().date()} → {df_final.index.max().date()}")
        print(f"Frequency: {freq}")
        for nombre in df_final.columns:
            print(f"{nombre}: {df_final[nombre].isna().sum()} NaNs")

    return df_final


def inferir_columna_fecha(df):
    """
    Infer datetime column from a DataFrame by dtype inspection.
    """
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    raise ValueError("No datetime column found.")
