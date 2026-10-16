"""
Network flow feature to audio signal conversion.

Core methodology: map normalized network features to sine wave frequencies
and superpose into a composite waveform representing the traffic flow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Audio parameters
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_DURATION = 2.0  # seconds
MIN_FREQ = 200.0   # Hz — audible lower bound
MAX_FREQ = 4000.0  # Hz — audible upper bound


class SignalGenerator:
    """
    Convert network flow feature vectors into audio signals.

    Each feature f_i is MinMax-scaled to [0, 1], mapped to a frequency
    in [MIN_FREQ, MAX_FREQ], and rendered as sin(2π f_i t). All feature
    waves are summed (superposition) into a single composite signal.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration: float = DEFAULT_DURATION,
        min_freq: float = MIN_FREQ,
        max_freq: float = MAX_FREQ,
        normalize_output: bool = True,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.normalize_output = normalize_output
        self.scaler = MinMaxScaler()
        self._is_fitted = False

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.duration)

    @property
    def time_axis(self) -> np.ndarray:
        return np.linspace(0, self.duration, self.n_samples, endpoint=False)

    def fit(self, X: np.ndarray) -> SignalGenerator:
        """Fit MinMax scaler on training feature matrix."""
        self.scaler.fit(X)
        self._is_fitted = True
        return self

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale a single feature vector or batch."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if self._is_fitted:
            return self.scaler.transform(features)
        return MinMaxScaler().fit_transform(features)

    def features_to_frequencies(self, features: np.ndarray) -> np.ndarray:
        """Map scaled features [0,1] to frequency range [min_freq, max_freq]."""
        scaled = self._scale_features(features)
        return self.min_freq + scaled * (self.max_freq - self.min_freq)

    def generate_component_waves(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate individual sine waves for each feature.

        Returns (frequencies, waves) where waves shape is (n_features, n_samples).
        """
        freqs = self.features_to_frequencies(features).flatten()
        t = self.time_axis
        waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs])
        return freqs, waves

    def generate_signal(self, features: np.ndarray) -> np.ndarray:
        """
        Generate superposition signal from feature vector.

        signal(t) = Σ_i sin(2π f_i t)
        """
        _, waves = self.generate_component_waves(features)
        signal = waves.sum(axis=0)

        if self.normalize_output:
            peak = np.max(np.abs(signal))
            if peak > 0:
                signal = signal / peak * 0.95

        return signal.astype(np.float32)

    def generate_batch(self, X: np.ndarray) -> np.ndarray:
        """Generate signals for a batch of feature vectors. Shape: (n, n_samples)."""
        return np.array([self.generate_signal(x) for x in X])

    def compute_fft(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute FFT magnitude spectrum of a signal."""
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        magnitude = np.abs(fft_vals)
        return freqs, magnitude

    def save_wav(
        self,
        signal: np.ndarray,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save signal as WAV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), signal, self.sample_rate)
        logger.debug("Saved WAV: %s", path)
        return path

    def save_batch_wavs(
        self,
        signals: np.ndarray,
        output_dir: str | Path,
        labels: np.ndarray | None = None,
        prefix: str = "flow",
    ) -> list[Path]:
        """Save batch of signals as individual WAV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, signal in enumerate(signals):
            label_str = f"_{labels[i]}" if labels is not None else ""
            path = output_dir / f"{prefix}_{i:05d}{label_str}.wav"
            self.save_wav(signal, path)
            paths.append(path)
        return paths

    def signal_to_spectrogram_image(
        self,
        signal: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Convert signal to log-mel spectrogram for CNN image models.
        Returns 2D array (n_mels, time_frames).
        """
        import librosa

        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128,
        )
        return librosa.power_to_db(mel, ref=np.max)
