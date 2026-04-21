from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

SEED = 42
META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
CLASS_ORDER_3 = ["NoPain", "PainArm", "PainHand"]
ARM_HAND = ("PainArm", "PainHand")


@dataclass
class ModelSpec:
    name: str
    params: dict


def load_clean_features(feature_parquet: str | Path) -> tuple[pd.DataFrame, list[str]]:
    fp = Path(feature_parquet)
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_parquet(fp)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    nan_frac = df[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if nan_frac[c] <= 0.10]
    med = df.loc[df["split"] == "train", feat_cols].median(numeric_only=True)
    X = df[feat_cols].copy().fillna(med)
    X = X.fillna(df[feat_cols].median(numeric_only=True)).fillna(0.0)
    var = X.loc[df["split"] == "train"].var(numeric_only=True)
    zero_var = var[var <= 1e-12].index.tolist()
    if zero_var:
        X = X.drop(columns=zero_var)
        feat_cols = [c for c in feat_cols if c not in zero_var]
    out = pd.concat([df[META_COLS].reset_index(drop=True), X[feat_cols].astype(np.float32).reset_index(drop=True)], axis=1)
    return out, feat_cols


def apply_subject_z(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    means = out.groupby("subject")[feat_cols].transform("mean")
    stds = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    stds = stds.where(stds > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - means) / stds).fillna(0.0).astype(np.float32)
    return out


def apply_subject_robust(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    med = out.groupby("subject")[feat_cols].transform("median")
    q75 = out.groupby("subject")[feat_cols].transform(lambda s: s.quantile(0.75))
    q25 = out.groupby("subject")[feat_cols].transform(lambda s: s.quantile(0.25))
    iqr = (q75 - q25).where((q75 - q25) > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - med) / iqr).fillna(0.0).astype(np.float32)
    return out


def build_norm_map(df_all: pd.DataFrame, feat_cols: list[str]) -> dict[str, pd.DataFrame]:
    return {
        "global": df_all.copy(),
        "subject_z": apply_subject_z(df_all, feat_cols),
        "subject_robust": apply_subject_robust(df_all, feat_cols),
    }


def pain_binary(classes: pd.Series) -> np.ndarray:
    return (classes != "NoPain").astype(int).to_numpy()


def armhand_binary(classes: pd.Series) -> np.ndarray:
    return classes.map({"PainArm": 0, "PainHand": 1}).to_numpy()


def class_codes_3(classes: pd.Series) -> np.ndarray:
    return classes.map({c: i for i, c in enumerate(CLASS_ORDER_3)}).to_numpy()


def channel_of(name: str) -> str:
    n = name.lower()
    hits = []
    for tag in ("bvp", "eda", "resp", "spo2", "spo_2"):
        if tag in n:
            hits.append("spo2" if tag == "spo_2" else tag)
    hits = sorted(set(hits))
    if not hits:
        return "other"
    if len(hits) == 1:
        return hits[0]
    return "cross"


def canonical_feature_name(name: str) -> str:
    return name[4:] if name.startswith("raw_") else name


def unique_by_canonical(features: list[str]) -> list[str]:
    seen = set()
    out = []
    ordered = sorted(features, key=lambda f: (f.startswith("raw_"), f))
    for feat in ordered:
        canon = canonical_feature_name(feat)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(feat)
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    diffs = a[:, None] - b[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / (a.size * b.size))


def rank_binary_features(train_df: pd.DataFrame, feat_cols: list[str], positive: str = "Pain") -> list[str]:
    tmp = train_df[["subject", "class"] + feat_cols].copy()
    if positive == "Pain":
        tmp["target"] = np.where(tmp["class"] == "NoPain", "NoPain", "Pain")
    else:
        tmp["target"] = np.where(tmp["class"] == "PainArm", "PainArm", "PainHand")
    agg = tmp.groupby(["subject", "target"])[feat_cols].mean(numeric_only=True)
    classes = agg.index.get_level_values("target").unique().tolist()
    if len(classes) < 2:
        return feat_cols
    a_df = agg.xs(classes[0], level="target")
    b_df = agg.xs(classes[1], level="target")
    common = a_df.index.intersection(b_df.index)
    a_df = a_df.loc[common]
    b_df = b_df.loc[common]
    scores = {}
    for feat in feat_cols:
        a = a_df[feat].to_numpy()
        b = b_df[feat].to_numpy()
        valid = ~(np.isnan(a) | np.isnan(b))
        scores[feat] = abs(cliffs_delta(a[valid], b[valid])) if valid.sum() >= 5 else 0.0
    return pd.Series(scores).sort_values(ascending=False).index.tolist()


def rank_multiclass_features(train_df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    agg = train_df.groupby(["subject", "class"])[feat_cols].mean(numeric_only=True)
    if agg.empty:
        return feat_cols
    present = [cls for cls in CLASS_ORDER_3 if cls in agg.index.get_level_values("class")]
    scores = {}
    for feat in feat_cols:
        groups = []
        for cls in present:
            values = agg.xs(cls, level="class")[feat].dropna().to_numpy(dtype=np.float64)
            if values.size:
                groups.append(values)
        total_n = sum(g.size for g in groups)
        if len(groups) < 2 or total_n <= len(groups):
            scores[feat] = 0.0
            continue
        grand_mean = float(np.concatenate(groups).mean())
        ss_between = sum(g.size * float((g.mean() - grand_mean) ** 2) for g in groups)
        ss_within = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
        if ss_between <= 1e-12:
            scores[feat] = 0.0
            continue
        if ss_within <= 1e-12:
            scores[feat] = float("inf")
            continue
        df_between = len(groups) - 1
        df_within = total_n - len(groups)
        scores[feat] = (ss_between / df_between) / (ss_within / df_within)
    return pd.Series(scores).sort_values(ascending=False).index.tolist()


EDA_BVP_FWD8 = [
    "Bvp_p90",
    "Eda_spec_spread",
    "Eda_energy",
    "Eda_hjorth_complexity",
    "Eda_bp_higher",
    "Eda_mean_abs",
    "BVP_env_std",
    "Bvp_bp_vhf_rel",
]


def stage1_feature_sets(train_df: pd.DataFrame, feat_cols: list[str]) -> dict[str, list[str]]:
    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    resp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "resp"])
    ranked_bvp_eda = rank_binary_features(train_df, bvp + eda, positive="Pain")
    ranked_resp = rank_binary_features(train_df, resp, positive="Pain")
    fwd8 = [f for f in EDA_BVP_FWD8 if f in feat_cols]
    return {
        "bvp_only": bvp,
        "eda_only": eda,
        "bvp_eda_core": bvp + eda,
        "bvp_eda_resp_small": ranked_bvp_eda[:60] + ranked_resp[:10],
        "eda_bvp_fwd8": fwd8,
        "all": unique_by_canonical(feat_cols),
        "all_raw": feat_cols,
    }


def stage2_feature_sets(train_df: pd.DataFrame, feat_cols: list[str]) -> dict[str, list[str]]:
    resp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "resp"])
    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    pain_df = train_df[train_df["class"].isin(ARM_HAND)].reset_index(drop=True)
    ranked_resp = rank_binary_features(pain_df, resp, positive="ArmHand")
    ranked_bvp = rank_binary_features(pain_df, bvp, positive="ArmHand")
    ranked_eda_resp = rank_binary_features(pain_df, eda + resp, positive="ArmHand")
    ranked_bvp_resp = rank_binary_features(pain_df, bvp + resp, positive="ArmHand")
    ranked_all = rank_binary_features(pain_df, bvp + eda + resp, positive="ArmHand")
    return {
        "resp_all": resp,
        "resp_top20": ranked_resp[:20],
        "resp_bvp5": ranked_resp[:20] + ranked_bvp[:5],
        "eda_resp_top30": ranked_eda_resp[:30],
        "bvp_resp_top30": ranked_bvp_resp[:30],
        "all_top40": ranked_all[:40],
    }


def multiclass_feature_sets(train_df: pd.DataFrame, feat_cols: list[str]) -> dict[str, list[str]]:
    all_modalities = unique_by_canonical(feat_cols)
    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    resp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "resp"])
    core = bvp + eda + resp
    ranked_core = rank_multiclass_features(train_df, core)
    ranked_all = rank_multiclass_features(train_df, all_modalities)
    return {
        "bvp_eda_resp": core,
        "top80": ranked_core[:80],
        "all_top120": ranked_all[:120],
        "all_modalities": all_modalities,
    }


def make_scaler(name: str):
    if name == "std":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(name)


def make_binary_model(spec: ModelSpec):
    name, p = spec.name, spec.params
    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 4)),
            learning_rate=float(p.get("learning_rate", 0.08)),
            subsample=float(p.get("subsample", 1.0)),
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            max_bin=int(p.get("max_bin", 128)),
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=4,
            verbosity=0,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 400)),
            max_depth=None if p.get("max_depth") in (None, "none") else int(p.get("max_depth")),
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "logreg":
        return LogisticRegression(
            C=float(p.get("C", 1.0)),
            class_weight="balanced",
            max_iter=4000,
            solver="lbfgs",
            random_state=SEED,
        )
    if name == "svm_linear":
        return SVC(kernel="linear", C=float(p.get("C", 1.0)), class_weight="balanced", probability=True, random_state=SEED)
    if name == "svm_rbf":
        gamma = p.get("gamma", "scale")
        if gamma not in ("scale", "auto"):
            gamma = float(gamma)
        return SVC(kernel="rbf", C=float(p.get("C", 1.0)), gamma=gamma, class_weight="balanced", probability=True, random_state=SEED)
    raise ValueError(name)


def make_multiclass_model(spec: ModelSpec, n_classes: int):
    name, p = spec.name, spec.params
    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 4)),
            learning_rate=float(p.get("learning_rate", 0.08)),
            subsample=float(p.get("subsample", 1.0)),
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            max_bin=int(p.get("max_bin", 128)),
            tree_method="hist",
            objective="multi:softprob",
            num_class=int(n_classes),
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=4,
            verbosity=0,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 400)),
            max_depth=None if p.get("max_depth") in (None, "none") else int(p.get("max_depth")),
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "logreg":
        return LogisticRegression(
            C=float(p.get("C", 1.0)),
            class_weight="balanced",
            max_iter=4000,
            solver="lbfgs",
            random_state=SEED,
        )
    if name == "svm_linear":
        return SVC(kernel="linear", C=float(p.get("C", 1.0)), class_weight="balanced", probability=True, random_state=SEED)
    if name == "svm_rbf":
        gamma = p.get("gamma", "scale")
        if gamma not in ("scale", "auto"):
            gamma = float(gamma)
        return SVC(kernel="rbf", C=float(p.get("C", 1.0)), gamma=gamma, class_weight="balanced", probability=True, random_state=SEED)
    raise ValueError(name)


def fit_binary_proba(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    y_train: np.ndarray,
    scaler_name: str,
    spec: ModelSpec,
    early_stopping: bool = False,
    es_rounds: int = 15,
    es_fraction: float = 0.15,
) -> np.ndarray:
    scaler = make_scaler(scaler_name)
    X_tr = train_df[feat_cols].to_numpy(dtype=np.float32)
    X_te = test_df[feat_cols].to_numpy(dtype=np.float32)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    if early_stopping and spec.name == "xgb" and HAS_XGB:
        from xgboost import XGBClassifier
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(X_tr))
        rng.shuffle(idx)
        cut = max(16, int((1 - es_fraction) * len(idx)))
        tr_i, va_i = idx[:cut], idx[cut:]
        params = spec.params
        mdl = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            max_bin=int(params.get("max_bin", 128)),
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=es_rounds,
            random_state=SEED,
            n_jobs=4,
            verbosity=0,
        )
        mdl.fit(X_tr[tr_i], y_train[tr_i], eval_set=[(X_tr[va_i], y_train[va_i])], verbose=False)
        return mdl.predict_proba(X_te).astype(np.float32)

    mdl = make_binary_model(spec)
    mdl.fit(X_tr, y_train)
    return mdl.predict_proba(X_te).astype(np.float32)


def fit_multiclass_proba(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str], y_train: np.ndarray, scaler_name: str, spec: ModelSpec, n_classes: int) -> np.ndarray:
    scaler = make_scaler(scaler_name)
    X_tr = train_df[feat_cols].to_numpy(dtype=np.float32)
    X_te = test_df[feat_cols].to_numpy(dtype=np.float32)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    mdl = make_multiclass_model(spec, n_classes=n_classes)
    mdl.fit(X_tr, y_train)
    probs = mdl.predict_proba(X_te)
    out = np.zeros((len(test_df), n_classes), dtype=np.float32)
    for i, cls_idx in enumerate(mdl.classes_):
        out[:, int(cls_idx)] = probs[:, i]
    return out


def fit_isotonic_binary(train_scores: np.ndarray, y_train: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.clip(train_scores, 1e-6, 1.0 - 1e-6), y_train)
    return lambda x: np.clip(iso.predict(np.asarray(x, dtype=np.float64)), 1e-6, 1.0 - 1e-6)


def fit_binary_calibrator(method: str, train_scores: np.ndarray, y_train: np.ndarray):
    scores = np.clip(np.asarray(train_scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y_train = np.asarray(y_train, dtype=np.int64)
    if method == "none":
        return lambda x: np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if method == "sigmoid":
        mdl = LogisticRegression(max_iter=4000, solver="lbfgs")
        mdl.fit(scores.reshape(-1, 1), y_train)
        return lambda x: mdl.predict_proba(np.clip(np.asarray(x), 1e-6, 1.0 - 1e-6).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return fit_isotonic_binary(scores, y_train)
    raise ValueError(method)


def exact_count_decode(log_scores: np.ndarray, target_counts: list[int]) -> np.ndarray:
    n_seg, n_classes = log_scores.shape
    dp = {tuple([0] * n_classes): 0.0}
    back = []
    for i in range(n_seg):
        next_dp = {}
        next_back = {}
        for state, score in dp.items():
            for cls_idx in range(n_classes):
                if state[cls_idx] >= target_counts[cls_idx]:
                    continue
                new_state = list(state)
                new_state[cls_idx] += 1
                new_state = tuple(new_state)
                new_score = score + float(log_scores[i, cls_idx])
                if new_score > next_dp.get(new_state, -np.inf):
                    next_dp[new_state] = new_score
                    next_back[new_state] = (state, cls_idx)
        dp = next_dp
        back.append(next_back)
    final_state = tuple(target_counts)
    if final_state not in dp:
        raise RuntimeError("exact-count decoding failed to find a feasible assignment")
    decoded = np.full(n_seg, -1, dtype=int)
    state = final_state
    for i in range(n_seg - 1, -1, -1):
        prev_state, cls_idx = back[i][state]
        decoded[i] = cls_idx
        state = prev_state
    return decoded


def exact12_binary_predictions(df_sub: pd.DataFrame, score_nopain: np.ndarray) -> np.ndarray:
    pred = np.ones(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-score_nopain[mask])]
        nop_idx = set(order[:12].tolist())
        pred[mask] = [0 if i in nop_idx else 1 for i in idx]
    return pred


def stage1_anchor_scores(df_eval: pd.DataFrame, df_eval_global: pd.DataFrame, feat_cols: list[str], score_nopain: np.ndarray, mode: str, lam: float) -> np.ndarray:
    out = np.zeros(len(df_eval), dtype=np.float32)
    for subject in sorted(df_eval["subject"].unique()):
        mask = (df_eval["subject"] == subject).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-score_nopain[mask])[:12]]
        base_ids = set(df_eval.iloc[order]["segment_id"].tolist())
        sub = df_eval_global[df_eval_global["subject"] == subject].reset_index(drop=True)
        base = sub["segment_id"].isin(base_ids)
        X = sub[feat_cols].to_numpy(dtype=np.float32)
        mu = sub.loc[base, feat_cols].mean(axis=0).to_numpy(dtype=np.float32)
        if mode == "center":
            dist = np.sqrt(np.mean((X - mu) ** 2, axis=1))
        elif mode == "z":
            sd = sub.loc[base, feat_cols].std(axis=0, ddof=0).to_numpy(dtype=np.float32)
            sd = np.where(sd > 1e-6, sd, 1.0)
            dist = np.mean(np.abs((X - mu) / sd), axis=1)
        else:
            raise ValueError(mode)
        p = score_nopain[mask].astype(np.float32)
        p_z = (p - p.mean()) / (p.std() + 1e-6)
        d_z = (dist - dist.mean()) / (dist.std() + 1e-6)
        out[mask] = p_z - lam * d_z
    return out


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_nopain": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_nopain": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }


def metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def decode_joint_weighted(stage1_probs: np.ndarray, stage2_probs: np.ndarray, w0: float = 0.8, w1: float = 1.2, w2: float = 1.4) -> np.ndarray:
    p_pain = np.clip(np.nan_to_num(stage1_probs[:, 1], nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6), 1e-6, 1.0 - 1e-6)
    p_arm = np.clip(np.nan_to_num(stage2_probs[:, 0], nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6), 1e-6, 1.0 - 1e-6)
    log_scores = np.column_stack([
        w0 * np.log1p(-p_pain),
        w1 * np.log(p_pain) + w2 * np.log(p_arm),
        w1 * np.log(p_pain) + w2 * np.log1p(-p_arm),
    ])
    log_scores = np.nan_to_num(log_scores, nan=-1e9, neginf=-1e9, posinf=1e9)
    return exact_count_decode(log_scores, [12, 12, 12])


def decode_split_topk(stage1_probs: np.ndarray, stage2_probs: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(stage1_probs), dtype=int)
    nop_idx = np.argsort(-stage1_probs[:, 0])[:12]
    pain_idx = np.array([i for i in range(len(stage1_probs)) if i not in set(nop_idx.tolist())])
    arm_idx = pain_idx[np.argsort(-stage2_probs[pain_idx, 0])[:12]]
    arm_set = set(arm_idx.tolist())
    nop_set = set(nop_idx.tolist())
    for i in range(len(stage1_probs)):
        if i in nop_set:
            pred[i] = 0
        elif i in arm_set:
            pred[i] = 1
        else:
            pred[i] = 2
    return pred


def plot_confusion(cm: np.ndarray, class_names: list[str], title: str, out_fp: str | Path) -> None:
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)


def plot_per_subject(subject_df: pd.DataFrame, title: str, out_fp: str | Path, value_col: str = "macro_f1") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    sub = subject_df.sort_values(value_col)
    ax.bar(sub["subject"].astype(str), sub[value_col], color="steelblue")
    ax.axhline(sub[value_col].mean(), color="red", linestyle="--", label=f"mean={sub[value_col].mean():.3f}")
    ax.set_title(title)
    ax.set_xlabel("subject")
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, prob: np.ndarray, title: str, out_fp: str | Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=8, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot(prob_pred, prob_true, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    fig.tight_layout()
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)


def ensure_parent(fp: str | Path) -> Path:
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    return fp
