import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model,
    train_sequences,
    train_labels,
    num_epochs,
    batch_size,
    learning_rate,
    patience=20,
    val_sequences=None,
    val_labels=None,
    device=None,
):
    """
    Train a sequence-to-sequence forecasting model with early stopping.

    The model must map inputs of shape (B, L, F) to predictions of shape
    (B, H, O). If H=1, outputs of shape (B, O) are automatically expanded
    to (B, 1, O).

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    train_sequences : torch.Tensor
        Training inputs of shape (B, L, F).
    train_labels : torch.Tensor
        Training targets of shape (B, H, O) or (B, O) if H=1.
    num_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Initial learning rate for Adam.
    patience : int, optional
        Early stopping patience.
    val_sequences : torch.Tensor, optional
        Validation inputs.
    val_labels : torch.Tensor, optional
        Validation targets.
    device : torch.device, optional
        Computation device.

    Returns
    -------
    model : nn.Module
        Model loaded with the best performing weights.
    history : list of dict
        Training history with keys {'train', 'val'} per epoch.
    """

    device = device or torch.device("cpu")
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(train_sequences, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = None
    if val_sequences is not None and val_labels is not None:
        val_loader = DataLoader(
            TensorDataset(val_sequences, val_labels),
            batch_size=batch_size,
            shuffle=False,
        )

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        # ===== Training =====
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            if y.dim() == 2:  # (B,O) → (B,1,O)
                y = y.unsqueeze(1)

            optimizer.zero_grad()
            y_hat = model(x)

            if y_hat.dim() == 2:
                y_hat = y_hat.unsqueeze(1)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # ===== Validation =====
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    if y.dim() == 2:
                        y = y.unsqueeze(1)

                    y_hat = model(x)
                    if y_hat.dim() == 2:
                        y_hat = y_hat.unsqueeze(1)

                    loss = criterion(y_hat, y)
                    val_loss += loss.item() * x.size(0)

            val_loss /= len(val_loader.dataset)
            monitor_loss = val_loss
        else:
            val_loss = None
            monitor_loss = train_loss

        history.append({"train": train_loss, "val": val_loss})

        # ===== Early stopping =====
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
