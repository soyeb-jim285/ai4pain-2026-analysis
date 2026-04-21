# Source Generated with Decompyle++
# File: 04_tfdomain_features.cpython-312.pyc (Python 3.12)

__doc__ = 'Time and frequency domain feature extraction for the AI4Pain 2026 dataset.\n\nOutputs\n-------\n- results/tables/tf_features.parquet\n- results/tables/tf_features_dictionary.csv\n- results/tables/tf_features_class_means.csv\n- results/tables/tf_features_anova_train.csv\n- plots/tfdomain/{psd_mean_per_class_*.png, band_power_bar_*.png,\n                  spectral_entropy_violin.png, hjorth_scatter.png,\n                  top16_tf_features_box.png}\n- results/reports/04_tfdomain_features_summary.md\n'
from __future__ import annotations
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matplotlib.pyplot import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sps
from scipy import stats as sstats
from tqdm import tqdm
from src.data_loader import CLASSES, SFREQ, SIGNALS, load_split
warnings.filterwarnings('ignore', category = RuntimeWarning)
# WARNING: Decompyle incomplete
