import numpy as np
import pandas as pd

from .garch import generar_sigma_multi_desde_backtesting
from .helpers import (
    calibrar_conformal_multi,
    cobertura_por_horizonte,
    _flatten_by_date,
)

def conformal_en_backtesting_split(
    backtesting_result: dict,
    series_original: pd.DataFrame,
    columnas_objetivo: list[str],
    *,
    alpha: float = 0.05,
    metodo: str = "estandarizado",   # "absoluto" | "estandarizado"
    h: int | None = None,
    mean: str = "Zero",
    vol: str = "EGARCH",
    p_g: int = 1, o_g: int = 0, q_g: int = 1,
    dist: str = "t",
    min_obs: int = 250,
    usar_retornos: bool = False,
    usar_iqr: bool = False,
    resolver: str = "mean",
    frac_cal: float = 0.7,
    ventana_garch: int | None = None,
):
    """
    Conformal prediction with an explicit calibration–evaluation split
    applied to rolling/expanding backtesting results.

    The function:
      1) Splits backtesting windows into calibration and evaluation sets.
      2) Estimates nonconformity scores (absolute or standardized via GARCH).
      3) Calibrates horizon- and variable-specific conformal quantiles.
      4) Builds prediction intervals on the evaluation windows only.
      5) Reports empirical coverage (global and per horizon).

    Parameters
    ----------
    backtesting_result : dict
        Output of `backtesting_multistep_flexible`.
    series_original : pd.DataFrame
        Original (unscaled) time series, used for volatility estimation.
    columnas_objetivo : list[str]
        Names of target variables.
    alpha : float
        Miscoverage level (e.g., 0.05 for 95% intervals).
    metodo : {"absoluto", "estandarizado"}
        Type of nonconformity score.
    h : int or None
        Maximum horizon to evaluate (defaults to full H).
    frac_cal : float
        Fraction of backtesting windows used for calibration.

    Returns
    -------
    low_flat_eval : pd.DataFrame
        Lower conformal bounds (evaluation set, flattened by date).
    high_flat_eval : pd.DataFrame
        Upper conformal bounds (evaluation set, flattened by date).
    cobertura_global_eval : pd.Series
        Average empirical coverage per variable.
    cobertura_h_eval : np.ndarray
        Empirical coverage by horizon and variable.
    q_info : dict
        Calibrated conformal quantiles per horizon and variable.
    """

    cols = list(columnas_objetivo)

    Yhat_win   = backtesting_result["panel"]["Yhat_win"]    # (N,H,D)
    Yreal_win  = backtesting_result["panel"]["Yreal_win"]   # (N,H,D)
    Fechas_win = backtesting_result["panel"]["Fechas_win"]  # (N,H)

    N, H_bt, D = Yhat_win.shape
    H = H_bt if h is None else min(h, H_bt)

    if not (0.0 < frac_cal < 1.0):
        raise ValueError("frac_cal must be in (0,1).")

    n_cal = int(np.floor(N * frac_cal))
    if n_cal < 5 or (N - n_cal) < 5:
        raise ValueError("Calibration/evaluation split too small.")

    idx_cal  = np.arange(n_cal)
    idx_eval = np.arange(n_cal, N)

    # ------------------------------------------------------------------
    # (1) Residuals (real scale)
    # ------------------------------------------------------------------
    residuos_multi = Yreal_win[:, :H, :] - Yhat_win[:, :H, :]  # (N,H,D)

    # ------------------------------------------------------------------
    # (2) Historical volatility (if standardized conformal)
    # ------------------------------------------------------------------
    if metodo == "estandarizado":
        sigma_hist_multi, _ = generar_sigma_multi_desde_backtesting(
            backtesting_result=backtesting_result,
            series_original=series_original[cols],
            columnas_objetivo=cols,
            h=H,
            mean=mean, vol=vol,
            p=p_g, o=o_g, q=q_g,
            dist=dist,
            min_obs=min_obs,
            usar_retornos=usar_retornos,
            ventana_garch=ventana_garch,
            show_warnings=False,
        )
    else:
        sigma_hist_multi = None

    # ------------------------------------------------------------------
    # (3) Calibration
    # ------------------------------------------------------------------
    residuos_cal = residuos_multi[idx_cal, :, :]

    if metodo == "estandarizado":
        if usar_retornos:
            Yhat_cal = Yhat_win[idx_cal, :H, :]
            sigma_hist_cal = (sigma_hist_multi[idx_cal] / 100.0) * np.maximum(Yhat_cal, 1e-12)
        else:
            sigma_hist_cal = sigma_hist_multi[idx_cal]
    else:
        sigma_hist_cal = None

    q_info = calibrar_conformal_multi(
        residuos_multi=residuos_cal,
        alpha=alpha,
        metodo=metodo,
        sigma_hist_multi=sigma_hist_cal,
        usar_iqr=usar_iqr,
        columnas=cols,
    )

    q_thr = np.asarray(q_info["q"], dtype=float)  # (H,D)

    # ------------------------------------------------------------------
    # (4) Prediction intervals on evaluation windows
    # ------------------------------------------------------------------
    Yhat_eval  = Yhat_win[idx_eval, :H, :]
    Yreal_eval = Yreal_win[idx_eval, :H, :]

    if metodo == "estandarizado":
        if usar_retornos:
            Yref = np.maximum(Yhat_eval, 1e-12)
            sigma_level = (sigma_hist_multi[idx_eval] / 100.0) * Yref
            low_win  = Yhat_eval - sigma_level * q_thr[None, :, :]
            high_win = Yhat_eval + sigma_level * q_thr[None, :, :]
        else:
            sigma_eval = sigma_hist_multi[idx_eval]
            low_win  = Yhat_eval - sigma_eval * q_thr[None, :, :]
            high_win = Yhat_eval + sigma_eval * q_thr[None, :, :]
    else:
        low_win  = Yhat_eval - q_thr[None, :, :]
        high_win = Yhat_eval + q_thr[None, :, :]

    # ------------------------------------------------------------------
    # (5) Coverage diagnostics
    # ------------------------------------------------------------------
    cobertura_h_eval = cobertura_por_horizonte(
        Yreal_win=Yreal_eval,
        Yhat_win=Yhat_eval,
        q_thr=q_thr,
        sigma_hist_multi=(
            sigma_hist_multi[idx_eval] if metodo == "estandarizado" else None
        ),
        usar_retornos=usar_retornos,
    )

    # ------------------------------------------------------------------
    # (6) Flatten by date (evaluation only)
    # ------------------------------------------------------------------
    fechas_eval = Fechas_win[idx_eval, :H]

    low_flat_eval  = _flatten_by_date(low_win,  fechas_eval, columnas=cols, resolver=resolver)
    high_flat_eval = _flatten_by_date(high_win, fechas_eval, columnas=cols, resolver=resolver)
    y_real_flat    = _flatten_by_date(Yreal_eval, fechas_eval, columnas=cols, resolver=resolver)

    inside = (
        (y_real_flat.values >= low_flat_eval.values) &
        (y_real_flat.values <= high_flat_eval.values)
    )

    cobertura_global_eval = pd.Series(inside.mean(axis=0), index=cols)

    return (
        low_flat_eval,
        high_flat_eval,
        cobertura_global_eval,
        cobertura_h_eval,
        q_info,
    )
