# Audio-Based Network Intrusion Detection System

Network flow records are converted into audio signals, then classified using standard audio feature extraction and machine learning. The project uses NSL-KDD as the primary dataset. CICIDS2017 is supported in code but not included in the result files below.

## Idea

Each feature in a flow record is scaled to [0, 1], mapped to a frequency between 200 and 4000 Hz, and written as a sine wave. The waves are summed into one signal per flow. Librosa extracts MFCCs, mel spectrograms, chroma, and related descriptors. Those vectors are compared against the original numeric features on the same labels.

## Repository layout

```
data/raw/              NSL-KDD files (download script provided)
notebooks/             Step-by-step analysis (01 through 06)
reports/               Metrics, logs, and figures
scripts/
  run_pipeline.py      One-command training and evaluation
  run_notebooks.py     Execute all notebooks in place
src/                   Core modules
```

## Setup

Requires Python 3.11 and an NVIDIA GPU for the deep learning steps (CPU works for baselines but will be slow).

```bash
git clone https://github.com/enigmatronix13/Audio-Based-Network-Intrusion-Detection-System.git
cd Audio-Based-Network-Intrusion-Detection-System

python -m venv venv
venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt shap lightgbm

python src/download_data.py
python scripts/run_pipeline.py
```

To re-run the notebooks with saved outputs:

```bash
python scripts/run_notebooks.py
```

## Results

All numbers in `reports/baseline_results.csv` and `reports/deep_learning_results.csv` come from `scripts/run_pipeline.py` on a stratified sample of real NSL-KDD data (1500 train, 1500 test). See `reports/RESULTS.md` for the full table and figure index.

Highlights:

| Model | Input | F1 |
|-------|-------|-----|
| XGBoost | network features | 0.775 |
| XGBoost | audio features | 0.750 |
| Random Forest | network features | 0.764 |
| ResNet18 | spectrogram (subset)* | 0.614 |

\* ResNet18 and EfficientNetB0 were trained on 800 flows and evaluated on 300 flows to keep runtime reasonable. MFCC models used the full 1500 test split.

Tree-based models perform similarly on tabular and audio features. The neural models need more data and tuning; details are in the results file.

## Figures

Plots live in `reports/figures/`. Most were produced when the notebooks were executed; the pipeline script writes a subset directly. A numbered index is in `reports/RESULTS.md`.

## License

MIT. See [LICENSE](LICENSE).
