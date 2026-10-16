"""
Data loading and preprocessing for NSL-KDD and CICIDS2017 datasets.

Provides unified interfaces for exploratory analysis, feature engineering,
and train/test splitting with reproducible pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# NSL-KDD column names (41 features + label + difficulty)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]

# Attack category mapping for NSL-KDD
NSL_KDD_ATTACK_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos", "teardrop": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
}

# CICIDS2017 attack label normalization
CICIDS_ATTACK_MAP = {
    "BENIGN": "normal",
    "Bot": "bot",
    "DDoS": "dos",
    "DoS GoldenEye": "dos",
    "DoS Hulk": "dos",
    "DoS Slowhttptest": "dos",
    "DoS slowloris": "dos",
    "FTP-Patator": "brute_force",
    "SSH-Patator": "brute_force",
    "Heartbleed": "dos",
    "Infiltration": "infiltration",
    "PortScan": "port_scan",
    "Web Attack - Brute Force": "brute_force",
    "Web Attack - Sql Injection": "web_attack",
    "Web Attack - XSS": "web_attack",
}


class DataLoader:
    """Load NSL-KDD and CICIDS2017 network flow datasets."""

    def __init__(self, data_dir: str | Path = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_nsl_kdd(
        self,
        train_file: str = "KDDTrain+.txt",
        test_file: str | None = "KDDTest+.txt",
        sample_size: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Load NSL-KDD train (and optional test) splits."""
        train_path = self.data_dir / train_file
        if not train_path.exists():
            logger.warning("NSL-KDD not found at %s. Generating synthetic sample.", train_path)
            return self._generate_synthetic_nsl_kdd(sample_size or 5000)

        train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
        train_df["label"] = train_df["label"].str.rstrip(".")
        train_df["attack_category"] = train_df["label"].map(NSL_KDD_ATTACK_MAP).fillna("other")
        train_df["is_attack"] = (train_df["label"] != "normal").astype(int)

        if sample_size:
            train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)

        test_df = None
        if test_file:
            test_path = self.data_dir / test_file
            if test_path.exists():
                test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
                test_df["label"] = test_df["label"].str.rstrip(".")
                test_df["attack_category"] = test_df["label"].map(NSL_KDD_ATTACK_MAP).fillna("other")
                test_df["is_attack"] = (test_df["label"] != "normal").astype(int)
                if sample_size:
                    test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

        logger.info("Loaded NSL-KDD: train=%d, test=%s", len(train_df), len(test_df) if test_df is not None else "N/A")
        return train_df, test_df

    def load_cicids2017(
        self,
        files: list[str] | None = None,
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        """Load and concatenate CICIDS2017 CSV files."""
        if files is None:
            files = sorted(self.data_dir.glob("*.csv"))
            files = [f.name for f in files if "MachineLearning" in f.name or "Friday" in f.name
                     or "Monday" in f.name or "Tuesday" in f.name or "Wednesday" in f.name
                     or "Thursday" in f.name]

        if not files:
            logger.warning("CICIDS2017 not found. Generating synthetic sample.")
            return self._generate_synthetic_cicids(sample_size or 5000)

        dfs = []
        for fname in files:
            path = self.data_dir / fname if isinstance(fname, str) else fname
            if Path(path).exists():
                chunk = pd.read_csv(path, low_memory=False)
                dfs.append(chunk)

        if not dfs:
            return self._generate_synthetic_cicids(sample_size or 5000)

        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        label_col = "Label" if "Label" in df.columns else "label"
        df["label"] = df[label_col].astype(str).str.strip()
        df["attack_category"] = df["label"].map(CICIDS_ATTACK_MAP).fillna("other")
        df["is_attack"] = (df["label"] != "BENIGN").astype(int)

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        logger.info("Loaded CICIDS2017: %d rows", len(df))
        return df

    def _generate_synthetic_nsl_kdd(self, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic NSL-KDD-like data for demo when real data is unavailable."""
        rng = np.random.default_rng(42)
        attacks = list(NSL_KDD_ATTACK_MAP.keys())
        labels = rng.choice(attacks, n, p=[0.4] + [0.6 / (len(attacks) - 1)] * (len(attacks) - 1))

        data = {}
        for col in NSL_KDD_COLUMNS[:-2]:
            if col in NSL_KDD_CATEGORICAL:
                if col == "protocol_type":
                    data[col] = rng.choice(["tcp", "udp", "icmp"], n)
                elif col == "service":
                    data[col] = rng.choice(["http", "smtp", "ftp", "ssh", "dns"], n)
                else:
                    data[col] = rng.choice(["SF", "S0", "REJ", "RSTO"], n)
            else:
                base = rng.exponential(1.0, n)
                attack_mask = labels != "normal"
                base[attack_mask] *= rng.uniform(2, 10, attack_mask.sum())
                data[col] = base

        data["label"] = labels
        data["difficulty"] = rng.integers(0, 21, n)
        df = pd.DataFrame(data)
        df["attack_category"] = df["label"].map(NSL_KDD_ATTACK_MAP).fillna("other")
        df["is_attack"] = (df["label"] != "normal").astype(int)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["is_attack"])
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _generate_synthetic_cicids(self, n: int) -> pd.DataFrame:
        """Generate synthetic CICIDS2017-like flow features."""
        rng = np.random.default_rng(42)
        labels = rng.choice(
            ["BENIGN", "DDoS", "PortScan", "FTP-Patator", "DoS Hulk"],
            n,
            p=[0.5, 0.15, 0.15, 0.1, 0.1],
        )

        df = pd.DataFrame({
            "Destination Port": rng.integers(1, 65535, n),
            "Flow Duration": rng.exponential(1000, n),
            "Total Fwd Packets": rng.poisson(10, n) + 1,
            "Total Backward Packets": rng.poisson(8, n),
            "Total Length of Fwd Packets": rng.exponential(500, n),
            "Total Length of Bwd Packets": rng.exponential(400, n),
            "Fwd Packet Length Max": rng.exponential(100, n),
            "Fwd Packet Length Min": rng.exponential(10, n),
            "Fwd Packet Length Mean": rng.exponential(50, n),
            "Fwd Packet Length Std": rng.exponential(20, n),
            "Bwd Packet Length Max": rng.exponential(100, n),
            "Flow Bytes/s": rng.exponential(1000, n),
            "Flow Packets/s": rng.exponential(100, n),
            "Flow IAT Mean": rng.exponential(50, n),
            "Flow IAT Std": rng.exponential(30, n),
            "Fwd IAT Total": rng.exponential(200, n),
            "Bwd IAT Total": rng.exponential(180, n),
            "SYN Flag Count": rng.integers(0, 5, n),
            "ACK Flag Count": rng.integers(0, 10, n),
            "Down/Up Ratio": rng.uniform(0, 2, n),
        })

        attack_mask = labels != "BENIGN"
        df.loc[attack_mask, "Flow Packets/s"] *= rng.uniform(3, 15, attack_mask.sum())
        df.loc[attack_mask, "SYN Flag Count"] += rng.integers(5, 50, attack_mask.sum())

        df["label"] = labels
        df["attack_category"] = df["label"].map(CICIDS_ATTACK_MAP).fillna("other")
        df["is_attack"] = (df["label"] != "BENIGN").astype(int)
        return df


class FeatureProcessor:
    """Feature engineering, encoding, scaling, and outlier handling."""

    def __init__(
        self,
        scaler_type: str = "minmax",
        handle_outliers: bool = True,
        outlier_std: float = 3.0,
    ):
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.outlier_std = outlier_std
        self.scaler: MinMaxScaler | StandardScaler | None = None
        self.label_encoder = LabelEncoder()
        self.feature_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self._encoders: dict[str, LabelEncoder] = {}

    def _get_scaler(self) -> MinMaxScaler | StandardScaler:
        if self.scaler_type == "standard":
            return StandardScaler()
        return MinMaxScaler()

    def identify_columns(self, df: pd.DataFrame, exclude: list[str] | None = None) -> None:
        """Auto-detect numeric and categorical feature columns."""
        exclude = exclude or ["label", "attack_category", "is_attack", "difficulty", "Label"]
        self.feature_columns = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        self.categorical_columns = [
            c for c in df.columns
            if c not in exclude and df[c].dtype == object and c not in self.feature_columns
        ]

    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return missing value counts and percentages per column."""
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).query("missing_count > 0")

    def detect_outliers(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Flag outliers using z-score method."""
        columns = columns or self.feature_columns
        outlier_flags = pd.DataFrame(index=df.index)
        for col in columns:
            if col not in df.columns:
                continue
            z = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-8))
            outlier_flags[f"{col}_outlier"] = (z > self.outlier_std).astype(int)
        outlier_flags["is_outlier"] = (outlier_flags.sum(axis=1) > 0).astype(int)
        return outlier_flags

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical columns."""
        df = df.copy()
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            if fit:
                self._encoders[col] = LabelEncoder()
                df[col] = self._encoders[col].fit_transform(df[col].astype(str))
            else:
                le = self._encoders[col]
                df[col] = df[col].astype(str).map(
                    {cls: i for i, cls in enumerate(le.classes_)}
                ).fillna(-1).astype(int)
        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        target_col: str = "is_attack",
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Full preprocessing pipeline: encode, impute, scale.

        Returns (X, y, feature_names).
        """
        if not self.feature_columns:
            self.identify_columns(df)

        work = df.copy()
        work = self.encode_categoricals(work, fit=fit)

        all_features = self.feature_columns + [
            c for c in self.categorical_columns if c in work.columns
        ]

        imputer = SimpleImputer(strategy="median")
        X_raw = work[all_features].values.astype(np.float64)

        if self.handle_outliers and fit:
            z = np.abs((X_raw - np.nanmean(X_raw, axis=0)) / (np.nanstd(X_raw, axis=0) + 1e-8))
            X_raw = np.where(z > self.outlier_std, np.nanmedian(X_raw, axis=0), X_raw)

        if fit:
            X_imputed = imputer.fit_transform(X_raw)
            self.scaler = self._get_scaler()
            X = self.scaler.fit_transform(X_imputed)
            self._imputer = imputer
        else:
            X_imputed = self._imputer.transform(X_raw)
            X = self.scaler.transform(X_imputed)

        y = work[target_col].values if target_col in work.columns else None
        return X, y, all_features

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train/test split."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def get_correlation_matrix(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Compute Pearson correlation matrix for numeric features."""
        columns = columns or self.feature_columns
        cols = [c for c in columns if c in df.columns]
        return df[cols].corr()

    def feature_importance_proxy(self, df: pd.DataFrame, target_col: str = "is_attack") -> pd.Series:
        """
        Compute univariate feature importance via absolute Pearson correlation
        with binary target (fast proxy for EDA).
        """
        if not self.feature_columns:
            self.identify_columns(df)
        work = df.copy()
        work = self.encode_categoricals(work, fit=True)
        cols = self.feature_columns + [c for c in self.categorical_columns if c in work.columns]
        corrs = {}
        for col in cols:
            if col in work.columns:
                val = work[col].corr(work[target_col])
                if pd.notna(val):
                    corrs[col] = abs(val)
        return pd.Series(corrs).sort_values(ascending=False)
