"""
Audio feature extraction from network-traffic-derived signals.

Uses librosa to extract standard audio descriptors that serve as
alternative feature representations for intrusion detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature configuration
N_MFCC = 20
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048


class AudioFeatureExtractor:
    """
    Extract comprehensive audio features from traffic signals.

    Features: MFCC, Mel Spectrogram stats, Chroma, Spectral Contrast,
    Tonnetz, Zero Crossing Rate, Spectral Centroid, RMS Energy.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = N_MFCC,
        n_mels: int = N_MELS,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        aggregate: str = "mean_std",
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.aggregate = aggregate  # 'mean_std', 'mean', 'flatten'

    def extract_all(self, signal: np.ndarray) -> dict[str, np.ndarray]:
        """Extract all audio features from a single signal."""
        y = signal.astype(np.float64)
        n_fft = min(self.n_fft, len(y))
        hop = min(self.hop_length, max(1, len(y) // 4))

        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc,
            hop_length=hop, n_fft=n_fft,
        )
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_mels=self.n_mels,
            hop_length=hop, n_fft=n_fft,
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sample_rate, hop_length=hop, n_fft=n_fft,
        )
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=self.sample_rate, hop_length=hop, n_fft=n_fft,
        )
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(y), sr=self.sample_rate,
        )
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, hop_length=hop, n_fft=n_fft,
        )
        rms = librosa.feature.rms(y=y, hop_length=hop)

        return {
            "mfcc": mfcc,
            "mel_spectrogram": mel_db,
            "chroma": chroma,
            "spectral_contrast": contrast,
            "tonnetz": tonnetz,
            "zcr": zcr,
            "spectral_centroid": centroid,
            "rms": rms,
        }

    def _aggregate_feature(self, feat: np.ndarray, name: str) -> dict[str, float]:
        """Aggregate 2D feature map to statistical vector."""
        result = {}
        if self.aggregate == "flatten":
            for i, val in enumerate(feat.flatten()):
                result[f"{name}_{i}"] = float(val)
        elif self.aggregate == "mean":
            agg = feat.mean(axis=1) if feat.ndim == 2 else feat
            for i, val in enumerate(agg):
                result[f"{name}_{i}"] = float(val)
        else:  # mean_std
            if feat.ndim == 2:
                means = feat.mean(axis=1)
                stds = feat.std(axis=1)
                for i, (m, s) in enumerate(zip(means, stds)):
                    result[f"{name}_{i}_mean"] = float(m)
                    result[f"{name}_{i}_std"] = float(s)
            else:
                result[f"{name}_mean"] = float(feat.mean())
                result[f"{name}_std"] = float(feat.std())
        return result

    def extract_feature_vector(self, signal: np.ndarray) -> np.ndarray:
        """Extract flat feature vector for ML models."""
        features = self.extract_all(signal)
        vec = []
        for name, feat in features.items():
            agg = self._aggregate_feature(feat, name)
            vec.extend(agg.values())
        return np.array(vec, dtype=np.float32)

    def extract_batch_vectors(self, signals: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        """Extract feature vectors for a batch of signals."""
        if len(signals) <= 32:
            return np.array([self.extract_feature_vector(s) for s in signals])
        try:
            from joblib import Parallel, delayed
            chunks = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self.extract_feature_vector)(s) for s in signals
            )
            return np.array(chunks, dtype=np.float32)
        except Exception:
            return np.array([self.extract_feature_vector(s) for s in signals])

    def extract_batch_mfcc(self, signals: np.ndarray) -> np.ndarray:
        """
        Extract MFCC sequences for deep learning.
        Returns shape (n_samples, n_mfcc, time_frames).
        """
        mfccs = []
        for signal in signals:
            mfcc = librosa.feature.mfcc(
                y=signal.astype(np.float64),
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
            )
            mfccs.append(mfcc)
        return np.array(mfccs, dtype=np.float32)

    def get_feature_names(self, signal: np.ndarray | None = None) -> list[str]:
        """Return ordered feature names matching extract_feature_vector output."""
        if signal is None:
            signal = np.random.randn(int(self.sample_rate * 2)).astype(np.float32)
        features = self.extract_all(signal)
        names = []
        for name, feat in features.items():
            names.extend(self._aggregate_feature(feat, name).keys())
        return names

    def features_to_dataframe(
        self,
        signals: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Extract features and return as DataFrame."""
        vectors = self.extract_batch_vectors(signals)
        names = self.get_feature_names(signals[0])
        df = pd.DataFrame(vectors, columns=names)
        if labels is not None:
            df["label"] = labels
        return df

    def get_dominant_frequencies(
        self,
        signal: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify dominant frequency components via FFT.
        Useful for explainability: which mapped frequencies matter most.
        """
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)
        magnitude = np.abs(fft)
        top_idx = np.argsort(magnitude)[-top_k:][::-1]
        return freqs[top_idx], magnitude[top_idx]

    def compare_attack_profiles(
        self,
        benign_signals: np.ndarray,
        attack_signals: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compare mean audio feature profiles between benign and attack traffic.
        Returns DataFrame with feature, benign_mean, attack_mean, difference.
        """
        benign_vecs = self.extract_batch_vectors(benign_signals)
        attack_vecs = self.extract_batch_vectors(attack_signals)
        names = self.get_feature_names(benign_signals[0])

        benign_mean = benign_vecs.mean(axis=0)
        attack_mean = attack_vecs.mean(axis=0)

        return pd.DataFrame({
            "feature": names,
            "benign_mean": benign_mean,
            "attack_mean": attack_mean,
            "difference": attack_mean - benign_mean,
            "abs_difference": np.abs(attack_mean - benign_mean),
        }).sort_values("abs_difference", ascending=False)
