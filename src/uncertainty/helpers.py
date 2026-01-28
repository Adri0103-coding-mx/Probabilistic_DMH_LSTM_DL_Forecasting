import numpy as np


def _flatten_by_date(errors_dict):
    """
    Convierte un dict {fecha: array[horizontes]}
    en una lista plana de errores (leakage-free).
    """
    flat = []
    for _, errs in errors_dict.items():
        flat.extend(np.asarray(errs).ravel())
    return np.asarray(flat)

def calibrar_conformal_multi(abs_errors, alpha):
    """
    abs_errors: np.array (flattened)
    alpha: nivel de significancia
    """
    return np.quantile(abs_errors, 1.0 - alpha, method="higher")

import numpy as np


def cobertura_por_horizonte(y_true, y_pred, q_hat):
    """
    y_true, y_pred: dict {h: np.array}
    q_hat: dict {h: escalar}
    """
    cobertura = {}

    for h in y_true:
        lower = y_pred[h] - q_hat[h]
        upper = y_pred[h] + q_hat[h]

        cobertura[h] = np.mean(
            (y_true[h] >= lower) & (y_true[h] <= upper)
        )

    return cobertura
