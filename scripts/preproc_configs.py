"""Preprocessing config registry for ablation sweep.

Each config is a PreprocConfig. Keep the list intentionally small (<= 20)
so the sweep stays tractable. Group by modality focus:
  - raw             : identity baseline
  - bvp_<variant>   : BVP-focused (other signals identity)
  - eda_<variant>   : EDA-focused (other signals identity)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import PreprocConfig  # noqa: E402


def _empty() -> dict[str, list]:
    return {"Bvp": [], "Eda": [], "Resp": [], "SpO2": []}


def _with(sig: str, ops: list) -> dict[str, list]:
    d = _empty()
    d[sig] = ops
    return d


REGISTRY: list[PreprocConfig] = [
    PreprocConfig(name="raw", ops=_empty(), notes="identity baseline"),

    PreprocConfig(
        name="bvp_butter_0p5_8",
        ops=_with("Bvp", [("butter_band", {"low": 0.5, "high": 8.0, "order": 3})]),
        notes="NK default pulse band",
    ),
    PreprocConfig(
        name="bvp_butter_0p5_5",
        ops=_with("Bvp", [("butter_band", {"low": 0.5, "high": 5.0, "order": 3})]),
    ),
    PreprocConfig(
        name="bvp_butter_0p8_3",
        ops=_with("Bvp", [("butter_band", {"low": 0.8, "high": 3.0, "order": 3})]),
        notes="narrow task band",
    ),
    PreprocConfig(
        name="bvp_nk_elgendi",
        ops=_with("Bvp", [("nk_ppg_clean", {"method": "elgendi"})]),
    ),
    PreprocConfig(
        name="bvp_butter_ma5",
        ops=_with(
            "Bvp",
            [
                ("butter_band", {"low": 0.5, "high": 8.0, "order": 3}),
                ("moving_average", {"window": 5}),
            ],
        ),
    ),
    PreprocConfig(
        name="bvp_wavelet",
        ops=_with("Bvp", [("wavelet_denoise", {"wavelet": "db4", "level": 4})]),
    ),

    # SOTA-inspired BVP configs
    PreprocConfig(
        name="bvp_butter_0p5_5_ord4",
        ops=_with("Bvp", [("butter_band", {"low": 0.5, "high": 5.0, "order": 4})]),
        notes="Subramaniam 2023 BVP pain (4th order Butter 0.5-5, 80% F1 XGB)",
    ),
    PreprocConfig(
        name="bvp_cheby2_0p5_10",
        ops=_with("Bvp", [("cheby2_band", {"low": 0.5, "high": 10.0, "order": 4, "rs": 40.0})]),
        notes="Cheby II 0.5-10 Hz, hypertension PPG standard",
    ),
    PreprocConfig(
        name="bvp_savgol_butter",
        ops=_with(
            "Bvp",
            [
                ("savgol", {"window": 11, "polyorder": 3}),
                ("butter_band", {"low": 0.5, "high": 8.0, "order": 3}),
            ],
        ),
        notes="Savitzky-Golay smoothing + Butter band",
    ),
    PreprocConfig(
        name="bvp_wavelet_butter",
        ops=_with(
            "Bvp",
            [
                ("wavelet_denoise", {"wavelet": "db4", "level": 4}),
                ("butter_band", {"low": 0.5, "high": 8.0, "order": 3}),
            ],
        ),
        notes="wavelet denoise + Butter band (pain-detection common)",
    ),
    PreprocConfig(
        name="bvp_butter_1_5",
        ops=_with("Bvp", [("butter_band", {"low": 1.0, "high": 5.0, "order": 3})]),
        notes="narrow HR-fundamental only",
    ),

    PreprocConfig(
        name="eda_butter_lp1",
        ops=_with("Eda", [("butter_low", {"high": 1.0, "order": 3})]),
    ),
    PreprocConfig(
        name="eda_butter_lp3",
        ops=_with("Eda", [("butter_low", {"high": 3.0, "order": 3})]),
    ),
    PreprocConfig(
        name="eda_butter_lp5",
        ops=_with("Eda", [("butter_low", {"high": 5.0, "order": 3})]),
    ),
    PreprocConfig(
        name="eda_nk_neurokit",
        ops=_with("Eda", [("nk_eda_clean", {"method": "neurokit"})]),
    ),
    PreprocConfig(
        name="eda_ma25",
        ops=_with("Eda", [("moving_average", {"window": 25})]),
    ),
    PreprocConfig(
        name="eda_detrend_lp1",
        ops=_with(
            "Eda",
            [
                ("detrend", {}),
                ("butter_low", {"high": 1.0, "order": 3}),
            ],
        ),
    ),

    # SOTA-inspired EDA configs (AI4Pain 2025, cvxEDA, pain-detection lit)
    PreprocConfig(
        name="eda_cvxeda_phasic",
        ops=_with("Eda", [("nk_eda_phasic", {"method": "cvxeda", "component": "phasic"})]),
        notes="cvxEDA phasic component (Greco 2015, 87% pain acc)",
    ),
    PreprocConfig(
        name="eda_cvxeda_both",
        ops=_with("Eda", [("nk_eda_phasic", {"method": "cvxeda", "component": "both"})]),
        notes="cvxEDA tonic+phasic reconstruction",
    ),
    PreprocConfig(
        name="eda_butter_lp0p2",
        ops=_with("Eda", [("butter_low", {"high": 0.2, "order": 3})]),
        notes="CrossMod-Transformer AI4Pain 2025 winning cutoff",
    ),
    PreprocConfig(
        name="eda_median_lp1",
        ops=_with(
            "Eda",
            [
                ("median", {"window": 5}),
                ("butter_low", {"high": 1.0, "order": 3}),
            ],
        ),
        notes="median + LP, common SOTA combo",
    ),
    PreprocConfig(
        name="bvp_nabian_eda_median_lp1",
        ops={
            "Bvp": [("nk_ppg_clean", {"method": "nabian2018"})],
            "Eda": [("median", {"window": 5}), ("butter_low", {"high": 1.0, "order": 3})],
            "Resp": [],
            "SpO2": [],
        },
        notes="bvp_pipeline winner (nabian LP-40Hz) + median_lp1 EDA",
    ),
    PreprocConfig(
        name="eda_hp0p05_lp1",
        ops=_with(
            "Eda",
            [
                ("butter_high", {"low": 0.05, "order": 3}),
                ("butter_low", {"high": 1.0, "order": 3}),
            ],
        ),
        notes="HP 0.05 (phasic isolate) + LP 1",
    ),
    PreprocConfig(
        name="eda_detrend_hp0p05_lp1",
        ops=_with(
            "Eda",
            [
                ("detrend", {}),
                ("butter_high", {"low": 0.05, "order": 3}),
                ("butter_low", {"high": 1.0, "order": 3}),
            ],
        ),
        notes="detrend (our winner) + SOTA HP+LP band",
    ),
]


def get_config(name: str) -> PreprocConfig:
    for c in REGISTRY:
        if c.name == name:
            return c
    raise KeyError(f"no config named {name!r}")


def dump_registry_json(fp):
    import json
    payload = [
        {"name": c.name, "ops": {k: list(v) for k, v in c.ops.items()}, "notes": c.notes}
        for c in REGISTRY
    ]
    json.dump(payload, fp, indent=2, default=str)
