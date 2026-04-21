# Source Generated with Decompyle++
# File: 03_physio_features.cpython-312.pyc (Python 3.12)

__doc__ = 'Physiological-feature extraction for the AI4Pain 2026 dataset.\n\nPer-segment (10 s @ 100 Hz) features across BVP, EDA, RESP, SpO2 plus\none cross-signal feature (BVP-RESP coupling proxy).\n\nOutputs tables, plots, and a summary report under results/ and plots/physio/.\n'
from __future__ import annotations
import sys
import warnings
from collections import Counter
from pathlib import Path
from matplotlib.pyplot import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as spsig
from scipy import stats as spstats
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import CLASSES, SFREQ, SIGNALS, load_split
warnings.filterwarnings('ignore')
# WARNING: Decompyle incomplete
