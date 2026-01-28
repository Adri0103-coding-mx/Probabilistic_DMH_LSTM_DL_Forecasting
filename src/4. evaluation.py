import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model,
    sequences,     # Tensor (N, L, F)
    labels,        # Tensor (N, O) or (N, H, O)
    scaler=None,   # Optional scaler fitted on labels
    h_idx=0,       # Horizon to evaluate (0 = t+1)
    batch_size=256
):
    """
    Evaluate a trained forecasting model on a fixed dataset.

    The model is expected to map inputs of shape (B, L, F) to outputs
    of shape (B, H, O). If H = 1, outputs of shape (B, O) are automatically
    expanded to (B, 1, O).

    Parameters
    ----------
    model : nn.Module
        Trained forecasting model.
    sequences : torch.Tensor
        Input sequences of shape (N, L, F).
    labels : torch.Tensor
        True targets of shape (N, O) or (N, H, O).
    scaler : object, optional
        Scaler used to inverse-transform predictions and labels.
    h_idx : int, optional
        Horizon index to evaluate (default: 0).
    batch_size : int, optional
        Batch size for inference.

    Returns
    -------
    metrics : dict
        Dictionary with MAE, RMSE and R2.
    y_true : np.ndarray
        True values at horizon h_idx, shape (N, O).
    y_pred : np.ndarray
        Predicted values at horizon h_idx, shape (N, O).
    """

    device = next(model.parameters()).device

    # --- Ensure labels are 3D: (N,H,O)
    if labels.dim() == 2:
        labels = labels.unsqueeze(1)

    N, H, O = labels.shape
    if not (0 <= h_idx < H):
        raise ValueError(f"h_idx={h_idx} out of range (H={H})")

    loader = DataLoader(
        TensorDataset(sequences, labels),
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()
    preds, truths = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            out = model(xb)
            if out.dim() == 2:
                out = out.unsqueeze(1)

            preds.append(out.cpu().numpy())
            truths.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)[:, h_idx, :]
    y_true = np.concatenate(truths, axis=0)[:, h_idx, :]

    # --- Inverse scaling if provided
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)

    # --- Metrics
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Horizon": h_idx + 1
    }

    return metrics, y_true, y_pred
