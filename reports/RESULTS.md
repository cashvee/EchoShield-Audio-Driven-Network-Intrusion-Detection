# Results

Generated on NSL-KDD with `python scripts/run_pipeline.py` unless noted otherwise. Hardware: NVIDIA GeForce RTX 4050 Laptop GPU. Runtime: about 14 minutes for the full pipeline.

Train/test split: 1500 flows each, stratified. Network features after encoding: 38. Audio feature vector length: 352.

## Baseline models

| Model | Features | Accuracy | F1 | ROC-AUC |
|-------|----------|----------|-----|---------|
| XGBoost | network | 0.787 | 0.775 | 0.915 |
| XGBoost | audio | 0.769 | 0.750 | 0.869 |
| Random Forest | network | 0.779 | 0.764 | 0.954 |
| Random Forest | audio | 0.764 | 0.744 | 0.870 |
| SVM | network | 0.745 | 0.723 | 0.906 |
| SVM | audio | 0.720 | 0.726 | 0.723 |
| Logistic Regression | network | 0.705 | 0.667 | 0.881 |
| Logistic Regression | audio | 0.665 | 0.608 | 0.630 |

Source: `baseline_results.csv`

## Deep learning

Full test set (1500 flows) for MFCC models. Spectrogram models use an 800/300 train/test subset.

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| ResNet18 | 0.547 | 0.614 | 0.552 |
| EfficientNetB0 | 0.507 | 0.513 | 0.496 |
| CNN+LSTM | 0.491 | 0.537 | 0.489 |
| LSTM | 0.494 | 0.474 | 0.512 |
| CNN1D | 0.475 | 0.335 | 0.486 |

Source: `deep_learning_results.csv`

Notebook runs (`05_deep_learning.ipynb`) use a smaller sample and will not match these numbers exactly.

## Figure index

| File | Description |
|------|-------------|
| 01_class_distribution.png | Class balance |
| 02_attack_distribution.png | Attack categories |
| 03_correlation_heatmap.png | Feature correlations |
| 04_feature_importance.png | Correlation with label |
| 05_outlier_boxplots.png | Box plots for top features |
| 06_pca_tsne.png | PCA and t-SNE |
| 07_umap.png | UMAP projection |
| 08_component_waves.png | Individual sine components |
| 09_combined_waveforms.png | Benign vs attack waveforms |
| 10_fft_spectrum.png | FFT comparison |
| 11_frequency_distribution.png | Mapped frequency histogram |
| 12_mfcc_heatmap.png | MFCC heatmaps |
| 13_mel_spectrogram.png | Mel spectrograms |
| 14_chroma_spectral.png | Chroma and spectral features |
| 15_baseline_comparison.png | F1: network vs audio (notebook run) |
| 16_confusion_matrix.png | Best DL model confusion matrix |
| 17_roc_curves.png | ROC curves (notebook DL run) |
| 18_training_curves.png | Training loss and accuracy |
| 19_shap_network_beeswarm.png | SHAP for network features |
| 20_shap_audio_beeswarm.png | SHAP for audio features |
| 21_frequency_importance.png | Top audio features by SHAP |
| 22_spectrogram_gallery.png | Spectrograms by attack type |
| 23_interactive_spectrogram.html | Plotly comparison (not tracked in git) |

## Reproduce

```bash
python src/download_data.py
python scripts/run_pipeline.py
python scripts/run_notebooks.py
```

Logs: `pipeline_run.log`, `notebooks_run.log` (local only, not committed).
