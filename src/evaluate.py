"""
Model evaluation utilities: metrics, comparison tables, and publication-quality plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Publication style
plt.style.use("seaborn-v0_8-whitegrid")
PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}


class Evaluator:
    """Comprehensive model evaluation and visualization."""

    def __init__(self, output_dir: str | Path = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(PUBLICATION_RC)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        average: str = "binary",
    ) -> dict[str, float]:
        """Compute standard classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        if y_prob is not None:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        return metrics

    def metrics_to_dataframe(self, results: dict[str, dict[str, float]]) -> pd.DataFrame:
        """Convert nested metrics dict to comparison DataFrame."""
        return pd.DataFrame(results).T.round(4)

    def compare_models(
        self,
        results_a: dict[str, dict[str, float]],
        results_b: dict[str, dict[str, float]],
        label_a: str = "Network Features",
        label_b: str = "Audio Features",
    ) -> pd.DataFrame:
        """Side-by-side comparison of two feature representation approaches."""
        df_a = self.metrics_to_dataframe(results_a)
        df_a["feature_type"] = label_a
        df_b = self.metrics_to_dataframe(results_b)
        df_b["feature_type"] = label_b
        combined = pd.concat([df_a, df_b])
        combined.index.name = "model"
        return combined.reset_index()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str] | None = None,
        title: str = "Confusion Matrix",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot annotated confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        labels = labels or ["Benign", "Attack"]

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    def plot_roc_curves(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
        title: str = "ROC Curves",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """
        Plot multiple ROC curves.

        curves: {model_name: (fpr, tpr, auc_score)}
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, (fpr, tpr, auc_score) in curves.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    def plot_precision_recall_curves(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray]],
        title: str = "Precision-Recall Curves",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot precision-recall curves for multiple models."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, (precision, recall) in curves.items():
            ax.plot(recall, precision, label=name, linewidth=2)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    def plot_training_curves(
        self,
        history: dict[str, list[float]],
        title: str = "Training History",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot training/validation loss and accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label="Train", linewidth=2)
            if "val_loss" in history:
                axes[0].plot(history["val_loss"], label="Validation", linewidth=2)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss")
            axes[0].legend()

        if "train_acc" in history:
            axes[1].plot(history["train_acc"], label="Train", linewidth=2)
            if "val_acc" in history:
                axes[1].plot(history["val_acc"], label="Validation", linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Accuracy")
            axes[1].legend()

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    def get_roc_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute FPR, TPR, and AUC for ROC plotting."""
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return fpr, tpr, auc(fpr, tpr)

    def get_pr_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute precision and recall arrays for PR curve."""
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return precision, recall

    def classification_report_df(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return classification report as DataFrame."""
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
        return pd.DataFrame(report).T.round(4)

    def plot_model_comparison_bar(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "f1",
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Bar chart comparing models on a single metric."""
        fig, ax = plt.subplots(figsize=(10, 5))
        if "feature_type" in comparison_df.columns:
            sns.barplot(
                data=comparison_df, x="model", y=metric,
                hue="feature_type", ax=ax,
            )
        else:
            sns.barplot(data=comparison_df, x=comparison_df.index, y=metric, ax=ax)

        ax.set_title(title or f"Model Comparison — {metric.upper()}")
        ax.set_ylabel(metric.upper())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    def save_comparison_table(
        self,
        comparison_df: pd.DataFrame,
        path: str | Path,
    ) -> Path:
        """Save comparison table to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(path, index=False)
        return path
