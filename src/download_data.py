"""Data download utilities for NSL-KDD and CICIDS2017 datasets."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

NSL_KDD_URLS = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
}

CICIDS2017_INFO = """
CICIDS2017 Dataset Download Instructions
=========================================
The CICIDS2017 dataset is ~2.8 GB and must be downloaded manually:

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the MachineLearningCSV.zip file
3. Extract CSV files to: data/raw/

Expected files:
  - Monday-WorkingHours.pcap_ISCX.csv
  - Tuesday-WorkingHours.pcap_ISCX.csv
  - Wednesday-workingHours.pcap_ISCX.csv
  - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  - Friday-WorkingHours-Morning.pcap_ISCX.csv
  - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

Alternatively, use the Kaggle mirror:
  https://www.kaggle.com/datasets/cicdataset/cicids2017
"""


def download_nsl_kdd(data_dir: str | Path = "data/raw") -> None:
    """Download NSL-KDD train and test files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in NSL_KDD_URLS.items():
        dest = data_dir / filename
        if dest.exists():
            logger.info("Already exists: %s", dest)
            continue
        logger.info("Downloading %s...", filename)
        try:
            urlretrieve(url, dest)
            logger.info("Saved: %s", dest)
        except Exception as e:
            logger.error("Failed to download %s: %s", filename, e)


def check_datasets(data_dir: str | Path = "data/raw") -> dict[str, bool]:
    """Check which datasets are available."""
    data_dir = Path(data_dir)
    status = {
        "nsl_kdd_train": (data_dir / "KDDTrain+.txt").exists(),
        "nsl_kdd_test": (data_dir / "KDDTest+.txt").exists(),
        "cicids2017": any(data_dir.glob("*.csv")),
    }
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_nsl_kdd()
    print(CICIDS2017_INFO)
    print("\nDataset status:", check_datasets())
