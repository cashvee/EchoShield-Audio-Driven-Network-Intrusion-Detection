Exploring network security through audio signal processing and machine learning.
EchoShield is a cybersecurity research project that explores an unconventional approach to Network Intrusion Detection Systems (NIDS): transforming network traffic records into audio signals and using audio-based representations to identify potentially malicious activity.
By combining signal processing, feature engineering, and machine learning, the project investigates whether network behavior can be represented acoustically while preserving information useful for distinguishing normal traffic from cyberattacks.
Traditional intrusion detection systems analyze network traffic through numerical features and packet-level information. EchoShield explores an alternative representation by mapping network flow features to sound and comparing the resulting audio-derived features with conventional network features.
The project uses the NSL-KDD dataset as its primary benchmark and supports experimentation with the CICIDS2017 dataset.
- 🎵 Network-to-Audio Transformation — Converts normalized network flow features into synthesized audio signals.
- 🔊 Audio Feature Extraction — Uses Librosa to extract MFCCs, mel spectrograms, chroma features, and other audio descriptors.
- 🤖 Machine Learning-Based Detection — Evaluates models using both conventional network features and audio-derived features.
- 🧠 Deep Learning Experiments — Explores neural network approaches for learning patterns from audio representations.
- 📊 Comparative Evaluation — Compares model performance across different feature representations using classification metrics.
- 📈 Reproducible Experiments — Includes notebooks, pipeline scripts, evaluation reports, and visualizations.
The system follows a structured processing pipeline:
1. Dataset Preparation: Load and preprocess network traffic records from NSL-KDD.
2. Feature Normalization: Scale network flow features to a common range.
3. Audio Synthesis: Map normalized feature values to frequencies and combine sine waves to generate an audio signal for each flow.
4. Audio Feature Extraction: Extract MFCCs, mel spectrograms, chroma, and related descriptors using Librosa.
5. Model Training: Train and evaluate machine learning and deep learning models using audio-based and conventional network features.
6. Performance Analysis: Compare results using metrics such as precision, recall, F1-score, and other evaluation measures.
- Language: Python
- Machine Learning: Scikit-learn, XGBoost, LightGBM
- Deep Learning: PyTorch, Torchvision
- Audio Processing: Librosa
- Data Processing: NumPy, Pandas
- Visualization: Matplotlib
- Environment: Jupyter Notebook
```
Audio-Based-Network-Intrusion-Detection-System/
├── data/
│   └── raw/             # Raw dataset files
├── notebooks/           # Analysis and experiments
├── reports/             # Results, metrics and figures
├── scripts/
│   ├── run_pipeline.py  # Training and evaluation pipeline
│   └── run_notebooks.py # Notebook execution utility
├── src/                 # Core implementation modules
├── requirements.txt
├── LICENSE
└── README.md
```
- Python 3.11
- pip
- NVIDIA GPU with compatible CUDA support recommended for deep learning experiments
Clone the repository:
```
git clone https://github.com/enigmatronix13/Audio-Based-Network-Intrusion-Detection-System.git
cd Audio-Based-Network-Intrusion-Detection-System
```
Create and activate a virtual environment:
```
python -m venv venv
```
On Windows:
```
venv\Scripts\activate
```
Install the dependencies:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt shap lightgbm
```
Download the dataset and run the pipeline:
```
python src/download_data.py
python scripts/run_pipeline.py
```
To execute the notebooks:
```
python scripts/run_notebooks.py
```
The project investigates the effectiveness of audio-derived representations for network intrusion detection by comparing them against conventional network features.
The repository's reported experiments include:
These are reported experimental results on sampled dataset splits, not guarantees of real-world detection performance. The deep learning spectrogram experiment used a smaller subset than the full test split.
The central research question is:
Can audio representations of network traffic provide useful features for detecting malicious network activity, and how do they compare with conventional numerical representations?
This project provides an experimental framework for investigating the intersection of audio signal processing, machine learning, and cybersecurity.
- Explore alternative network-to-audio encoding strategies.
- Evaluate additional audio feature representations and neural architectures.
- Investigate generalization across datasets and unseen attack categories.
- Study computational overhead and the feasibility of real-time detection.
- Improve robustness against class imbalance and evolving attack patterns.
This is an experimental research project based on benchmark datasets. Its results should not be interpreted as evidence of production-ready or real-time intrusion detection without further testing on representative network traffic.
This project is distributed under the MIT License. See the LICENSE file for details.
Investigating cybersecurity from a different signal: the sound of network traffic.
