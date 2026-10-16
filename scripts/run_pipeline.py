"""
End-to-end pipeline runner.

Executes the full research workflow and writes figures + metrics to reports/.
Run from project root:  python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torch
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audio_features import AudioFeatureExtractor
from evaluate import Evaluator
from models import (
    CNN1DClassifier, LSTMClassifier, CNNLSTMHybrid, SpectrogramCNN,
    ModelTrainer, create_dataloaders, DEVICE,
)
from preprocessing import DataLoader, FeatureProcessor
from signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
SAMPLE_SIZE = 1500
RANDOM_STATE = 42
DL_EPOCHS = 12


def setup_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "signals").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(ROOT / "data" / "raw")
    train_df, test_df = loader.load_nsl_kdd(sample_size=SAMPLE_SIZE)
    if test_df is None:
        train_df, test_df = train_test_split(
            train_df, test_size=0.2, random_state=RANDOM_STATE, stratify=train_df["is_attack"]
        )
    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_eda(train_df: pd.DataFrame, processor: FeatureProcessor) -> None:
    logger.info("Generating EDA figures...")
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = train_df["is_attack"].value_counts()
    axes[0].bar(["Benign", "Attack"], [counts.get(0, 0), counts.get(1, 0)], color=["#4c72b0", "#c44e52"])
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Count")

    attacks = train_df["attack_category"].value_counts()
    axes[1].barh(attacks.index, attacks.values)
    axes[1].set_title("Attack categories")
    axes[1].set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    processor.identify_columns(train_df)
    top = processor.feature_columns[:15]
    corr = processor.get_correlation_matrix(train_df, top)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, ax=ax)
    ax.set_title("Feature correlation (top 15 numeric features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    importance = processor.feature_importance_proxy(train_df)
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.head(15).plot(kind="barh", ax=ax)
    ax.set_title("Feature correlation with attack label")
    ax.set_xlabel("|Pearson r|")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_signal_generation(X: np.ndarray, y: np.ndarray, sig_gen: SignalGenerator) -> np.ndarray:
    logger.info("Generating audio signals...")
    signals = sig_gen.generate_batch(X)

    benign_idx = np.where(y == 0)[0][0]
    attack_idx = np.where(y == 1)[0][0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = sig_gen.time_axis
    axes[0].plot(t, signals[benign_idx], color="#4c72b0", linewidth=0.7)
    axes[0].set_title("Benign flow waveform")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(t, signals[attack_idx], color="#c44e52", linewidth=0.7)
    axes[1].set_title("Attack flow waveform")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_waveforms.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, idx, title in [
        (axes[0], benign_idx, "Benign FFT"),
        (axes[1], attack_idx, "Attack FFT"),
    ]:
        freqs, mag = sig_gen.compute_fft(signals[idx])
        ax.plot(freqs, mag, linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, 5000)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_fft_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close()

    sig_gen.save_batch_wavs(signals[:20], ROOT / "data" / "signals", labels=y[:20], prefix="flow")
    return signals


def run_audio_features(signals: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, AudioFeatureExtractor]:
    logger.info("Extracting audio features...")
    import librosa
    import librosa.display

    extractor = AudioFeatureExtractor()
    benign_s, attack_s = signals[y == 0][0], signals[y == 1][0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sig, title in [(axes[0], benign_s, "Benign mel spectrogram"), (axes[1], attack_s, "Attack mel spectrogram")]:
        mel = librosa.feature.melspectrogram(y=sig, sr=22050, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=22050, ax=ax, cmap="magma")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_mel_spectrograms.png", dpi=200, bbox_inches="tight")
    plt.close()

    vectors = extractor.extract_batch_vectors(signals, n_jobs=4)
    logger.info("Extracted audio features: %s", vectors.shape)
    pd.DataFrame(vectors, columns=extractor.get_feature_names(signals[0])).assign(label=y).to_csv(
        ROOT / "data" / "processed" / "audio_features.csv", index=False
    )
    return vectors, extractor


def run_baselines(
    X_train, y_train, X_test, y_test,
    X_train_audio, X_test_audio,
) -> pd.DataFrame:
    logger.info("Training baseline models...")
    evaluator = Evaluator(FIG_DIR)
    xgb_params = dict(
        n_estimators=100, random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0,
        tree_method="hist",
        device="cuda" if DEVICE.type == "cuda" else "cpu",
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(**xgb_params),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        m = evaluator.compute_metrics(y_test, y_pred, y_prob)
        rows.append({"model": name, "feature_type": "network", **m})

        if name == "XGBoost":
            model_audio = XGBClassifier(**xgb_params)
        else:
            model_audio = type(model)(**model.get_params())
        model_audio.fit(X_train_audio, y_train)
        y_pred2 = model_audio.predict(X_test_audio)
        y_prob2 = model_audio.predict_proba(X_test_audio)
        m2_metrics = evaluator.compute_metrics(y_test, y_pred2, y_prob2)
        rows.append({"model": name, "feature_type": "audio", **m2_metrics})
        logger.info("%s — network F1: %.3f | audio F1: %.3f", name, m["f1"], m2_metrics["f1"])

    df = pd.DataFrame(rows).round(4)
    df.to_csv(REPORT_DIR / "baseline_results.csv", index=False)

    pivot = df.pivot(index="model", columns="feature_type", values="f1")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, rot=45)
    ax.set_title("F1 score: network features vs audio features")
    ax.set_ylabel("F1")
    ax.legend(title="Feature type")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    return df


def run_deep_learning(
    mfcc_train, y_train, mfcc_test, y_test,
    sig_gen: SignalGenerator, X_train: np.ndarray, X_test: np.ndarray,
) -> pd.DataFrame:
    logger.info("Training deep learning models on %s...", DEVICE)
    evaluator = Evaluator(FIG_DIR)

    X_tr, X_val, y_tr, y_val = train_test_split(
        mfcc_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    results = {}
    histories = {}
    test_loader, _ = create_dataloaders(
        mfcc_test, y_test, mfcc_test[:1], y_test[:1], batch_size=64, dataset_type="mfcc"
    )

    for name, model in [
        ("CNN1D", CNN1DClassifier(n_mfcc=20)),
        ("LSTM", LSTMClassifier(input_size=20)),
        ("CNN+LSTM", CNNLSTMHybrid(n_mfcc=20)),
    ]:
        train_loader, val_loader = create_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=64, dataset_type="mfcc")
        trainer = ModelTrainer(model, device=DEVICE)
        histories[name] = trainer.fit(train_loader, val_loader, epochs=DL_EPOCHS)
        y_pred, y_prob = trainer.predict(test_loader)
        results[name] = evaluator.compute_metrics(y_test, y_pred, y_prob)
        trainer.save(ROOT / "models" / f"{name.lower().replace('+', '_')}.pt")
        logger.info("%s — F1: %.3f", name, results[name]["f1"])

    evaluator.plot_training_curves(histories["CNN1D"], title="CNN1D training", save_path=FIG_DIR / "08_training_curves.png")

    # Spectrogram CNNs (subset for transfer-learning speed)
    n_spec = min(800, len(X_train))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), n_spec, replace=False)
    X_spec = X_train[idx]
    y_spec = y_train[idx]

    def to_specs(X):
        return np.array([sig_gen.signal_to_spectrogram_image(sig_gen.generate_signal(x)) for x in X])

    spec_all = to_specs(X_spec)
    spec_mean, spec_std = spec_all.mean(), spec_all.std()
    spec_all = (spec_all - spec_mean) / (spec_std + 1e-8)

    X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(
        spec_all, y_spec, test_size=0.2, random_state=RANDOM_STATE, stratify=y_spec
    )
    n_test = min(300, len(X_test))
    spec_test = (to_specs(X_test[:n_test]) - spec_mean) / (spec_std + 1e-8)
    y_spec_test = y_test[:n_test]

    for name, backbone in [("ResNet18", "resnet18"), ("EfficientNetB0", "efficientnet_b0")]:
        train_loader, val_loader = create_dataloaders(X_tr_s, y_tr_s, X_val_s, y_val_s, batch_size=32, dataset_type="spec")
        test_loader_s, _ = create_dataloaders(spec_test, y_spec_test, spec_test[:1], y_spec_test[:1], batch_size=32, dataset_type="spec")

        model = SpectrogramCNN(backbone=backbone, n_classes=2, pretrained=True)
        trainer = ModelTrainer(model, device=DEVICE)
        histories[name] = trainer.fit(train_loader, val_loader, epochs=DL_EPOCHS)
        y_pred, y_prob = trainer.predict(test_loader_s)
        results[name] = evaluator.compute_metrics(y_spec_test, y_pred, y_prob)
        trainer.save(ROOT / "models" / f"{name.lower()}.pt")
        logger.info("%s — F1: %.3f", name, results[name]["f1"])

    roc_data = {}
    for name, fname in [("CNN1D", "cnn1d"), ("LSTM", "lstm")]:
        m = CNN1DClassifier(n_mfcc=20) if name == "CNN1D" else LSTMClassifier(input_size=20)
        t = ModelTrainer(m, device=DEVICE)
        t.load(ROOT / "models" / f"{fname}.pt")
        _, ypr = t.predict(test_loader)
        roc_data[name] = evaluator.get_roc_data(y_test, ypr)
    evaluator.plot_roc_curves(roc_data, save_path=FIG_DIR / "10_roc_curves.png")
    plt.close("all")

    df = pd.DataFrame(results).T.round(4)
    df.to_csv(REPORT_DIR / "deep_learning_results.csv")
    return df



def main() -> None:
    t0 = time.time()
    setup_dirs()
    logger.info("Compute device: %s", DEVICE)
    if DEVICE.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    train_df, test_df = load_data()
    processor = FeatureProcessor()
    processor.identify_columns(train_df)
    run_eda(train_df, processor)

    X_train, y_train, feat_names = processor.prepare_features(train_df, fit=True)
    X_test, y_test, _ = processor.prepare_features(test_df, fit=False)

    sig_gen = SignalGenerator()
    sig_gen.fit(X_train)
    train_signals = run_signal_generation(X_train, y_train, sig_gen)
    test_signals = sig_gen.generate_batch(X_test)

    X_train_audio, extractor = run_audio_features(train_signals, y_train)
    X_test_audio = extractor.extract_batch_vectors(test_signals, n_jobs=4)
    logger.info("Test audio features: %s", X_test_audio.shape)
    audio_names = extractor.get_feature_names(train_signals[0])

    baseline_df = run_baselines(X_train, y_train, X_test, y_test, X_train_audio, X_test_audio)

    mfcc_train = extractor.extract_batch_mfcc(train_signals)
    mfcc_test = extractor.extract_batch_mfcc(test_signals)
    dl_df = run_deep_learning(mfcc_train, y_train, mfcc_test, y_test, sig_gen, X_train, X_test)

    # SHAP
    try:
        import shap
        n = min(150, len(X_test))
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), n, replace=False)
        rf_net = RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1)
        rf_aud = RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1)
        rf_net.fit(X_train, y_train)
        rf_aud.fit(X_train_audio, y_train)

        explainer = shap.TreeExplainer(rf_aud)
        sv = explainer.shap_values(X_test_audio[idx])
        sv_plot = sv[1] if isinstance(sv, list) else sv
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv_plot, X_test_audio[idx], feature_names=audio_names, max_display=15, show=False)
        plt.title("SHAP — audio features (attack class)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "09_shap_audio.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("SHAP figure saved")
    except Exception as e:
        logger.warning("SHAP step skipped: %s", e)

    summary = {
        "sample_size_train": len(train_df),
        "sample_size_test": len(test_df),
        "n_network_features": int(X_train.shape[1]),
        "n_audio_features": int(X_train_audio.shape[1]),
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "runtime_seconds": round(time.time() - t0, 1),
        "baseline_top_f1_network": float(baseline_df[baseline_df["feature_type"] == "network"]["f1"].max()),
        "baseline_top_f1_audio": float(baseline_df[baseline_df["feature_type"] == "audio"]["f1"].max()),
    }
    (REPORT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Done in %.1fs. Figures: %s", summary["runtime_seconds"], FIG_DIR)
    logger.info("Reports: %s", REPORT_DIR)


if __name__ == "__main__":
    main()
