"""
Deep learning models for audio-based intrusion detection.

Architectures: CNN1D (MFCC), LSTM (sequential), CNN+LSTM hybrid,
and Spectrogram CNN with transfer learning (ResNet18, EfficientNetB0).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MFCCDataset(Dataset):
    """Dataset for MFCC tensors (n_samples, n_mfcc, time)."""

    def __init__(self, mfcc: np.ndarray, labels: np.ndarray):
        self.mfcc = torch.FloatTensor(mfcc)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mfcc[idx], self.labels[idx]


class SpectrogramDataset(Dataset):
    """Dataset for 2D spectrogram images (n_samples, 1, H, W)."""

    def __init__(self, specs: np.ndarray, labels: np.ndarray):
        if specs.ndim == 3:
            specs = specs[:, np.newaxis, :, :]
        self.specs = torch.FloatTensor(specs)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.specs[idx], self.labels[idx]


class CNN1DClassifier(nn.Module):
    """
    1D CNN for MFCC-based intrusion detection.

    Input: (batch, n_mfcc, time_frames)
    """

    def __init__(
        self,
        n_mfcc: int = 20,
        n_classes: int = 2,
        channels: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        channels = channels or [32, 64, 128]
        layers: list[nn.Module] = []

        in_ch = n_mfcc
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LSTMClassifier(nn.Module):
    """
    LSTM for sequential audio feature classification.

    Input: (batch, seq_len, n_features) — transposed MFCC or feature sequences.
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features, time) -> (batch, time, features)
        if x.dim() == 3 and x.size(1) != x.size(2):
            x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.classifier(h)


class CNNLSTMHybrid(nn.Module):
    """Hybrid CNN + LSTM model for MFCC sequences."""

    def __init__(
        self,
        n_mfcc: int = 20,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x)  # (batch, channels, time)
        cnn_out = cnn_out.transpose(1, 2)  # (batch, time, channels)
        _, (h_n, _) = self.lstm(cnn_out)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)


class SpectrogramCNN(nn.Module):
    """
    Transfer learning CNN for spectrogram images.

    Supports ResNet18 and EfficientNetB0 backbones via torchvision.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        n_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Linear(in_features, n_classes)
        elif backbone == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, n_classes),
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze classifier head
            if backbone == "resnet18":
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
            else:
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ModelTrainer:
    """Training loop with history tracking for deep learning models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device or DEVICE)
        self.device = device or DEVICE
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool = True,
    ) -> tuple[float, float]:
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(y_batch)
                correct += (outputs.argmax(1) == y_batch).sum().item()
                total += len(y_batch)

        return total_loss / total, correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 20,
        early_stopping_patience: int = 5,
    ) -> dict[str, list[float]]:
        """Train model with optional validation and early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if val_loader:
                val_loss, val_acc = self._run_epoch(val_loader, train=False)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

            if (epoch + 1) % 5 == 0:
                msg = f"Epoch {epoch+1}/{epochs} — loss: {train_loss:.4f}, acc: {train_acc:.4f}"
                if val_loader:
                    msg += f", val_loss: {self.history['val_loss'][-1]:.4f}"
                logger.info(msg)

        return self.history

    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions and probabilities."""
        self.model.eval()
        all_preds, all_probs = [], []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = F.softmax(outputs, dim=1)
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }, path)

    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    dataset_type: str = "mfcc",
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    DatasetClass = MFCCDataset if dataset_type == "mfcc" else SpectrogramDataset
    train_ds = DatasetClass(X_train, y_train)
    val_ds = DatasetClass(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
