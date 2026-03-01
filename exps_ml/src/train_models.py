\
import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE

import shap
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunConfig:
    # Inputs
    data_path: Path = PROJECT_ROOT / "imputed_data.csv"
    exposure_map_path: Path = PROJECT_ROOT / "configs" / "exposures_by_disease.csv"

    # Reproducibility
    random_state: int = 42

    # Feature selection
    n_features_select: int = 20

    # Grid-tuning (done once per disease)
    test_size_grid: float = 0.2

    # Repeated CV
    n_iterations: int = 10
    n_folds: int = 5

    # Demographics (preferred: by column name)
    demo_cols: Tuple[str, ...] = ("age", "gender", "race", "education", "PIR", "eGFR", "log_urinary_creatinine")
    demo_categorical_cols: Tuple[str, ...] = ("gender", "race", "education")

    # Outputs
    out_dir: Path = PROJECT_ROOT / "outputs"
    plot_model: str = "LightGBM"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_exposure_mapping(csv_path: Path) -> Dict[str, List[str]]:
    m = pd.read_csv(csv_path)
    need = {"disease", "exposure"}
    if not need.issubset(m.columns):
        raise ValueError(f"Exposure map must contain columns: {need}")

    m = m.dropna(subset=["disease", "exposure"]).copy()
    m["disease"] = m["disease"].astype(str).str.strip()
    m["exposure"] = m["exposure"].astype(str).str.strip()

    return m.groupby("disease")["exposure"].apply(list).to_dict()


def build_demo_features(df: pd.DataFrame, demo_cols: Tuple[str, ...], categorical_cols: Tuple[str, ...]) -> pd.DataFrame:
    missing = [c for c in demo_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing demographics columns in data: "
            f"{missing}. Update RunConfig.demo_cols to match your dataset."
        )

    demo = df.loc[:, list(demo_cols)].copy()

    demo_encoded = pd.get_dummies(
        demo,
        columns=[c for c in categorical_cols if c in demo.columns],
        drop_first=True
    )

    bool_cols = demo_encoded.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        demo_encoded[bool_cols] = demo_encoded[bool_cols].astype(int)

    return demo_encoded.reset_index(drop=True)


def select_features_lgbm_shap(X: pd.DataFrame, y: pd.Series, n_features: int, random_state: int = 42) -> List[str]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = lgb.LGBMClassifier(random_state=random_state)
    model.fit(X_scaled, y)

    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    if shap_values.values.ndim == 3:
        shap_vals = shap_values.values[:, :, 1]  # class=1
    else:
        shap_vals = shap_values.values

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-n_features:][::-1]
    return X.columns[top_idx].tolist()


def get_model_configs(random_state: int = 42) -> Dict[str, Dict]:
    return {
        "LightGBM": {
            "estimator": LGBMClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
            },
        },
        "XGBoost": {
            "estimator": xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state
            ),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
            },
        },
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def tune_hyperparams_once(
    X: pd.DataFrame,
    y: pd.Series,
    model_cfgs: Dict[str, Dict],
    test_size: float,
    random_state: int
) -> Dict[str, Dict]:
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    best_params = {}
    for name, cfg in model_cfgs.items():
        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            scoring="roc_auc",
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_res_scaled, y_res)
        best_params[name] = grid.best_params_

    return best_params


def repeated_stratified_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_ctor,
    best_params: Dict,
    n_iterations: int,
    n_folds: int
) -> pd.DataFrame:
    rows = []

    for it in range(n_iterations):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=it)

        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            smote = SMOTE(random_state=it)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            scaler = StandardScaler()
            X_res_scaled = scaler.fit_transform(X_res)
            X_test_scaled = scaler.transform(X_test)

            clf = model_ctor(**best_params)
            clf.fit(X_res_scaled, y_res)

            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            m = compute_metrics(y_test.values, y_prob)

            fold_rows.append({
                "Iteration": it + 1,
                "Fold": fold,
                **m
            })

        it_df = pd.DataFrame(fold_rows)
        it_mean = it_df.mean(numeric_only=True).to_dict()
        rows.append({
            "Model": model_name,
            "Iteration": it + 1,
            **it_mean
        })

    return pd.DataFrame(rows)


def run_for_one_disease(df: pd.DataFrame, disease: str, exposure_map: Dict[str, List[str]], cfg: RunConfig) -> pd.DataFrame:
    if disease not in exposure_map:
        raise KeyError(f"Disease '{disease}' not found in exposure mapping file.")

    df_d = df.dropna(subset=[disease]).reset_index(drop=True)
    y = df_d[disease].replace({"Yes": 1, "No": 0}).astype(int)

    exposure_cols = exposure_map[disease]
    missing = [c for c in exposure_cols if c not in df_d.columns]
    if missing:
        raise KeyError(f"[{disease}] Missing exposure columns in data: {missing}")

    X_exps = df_d[exposure_cols].copy()
    X_demo = build_demo_features(df_d, cfg.demo_cols, cfg.demo_categorical_cols)

    selected_exps = select_features_lgbm_shap(
        X_exps, y,
        n_features=cfg.n_features_select,
        random_state=cfg.random_state
    )

    X_expo_demo = pd.concat([X_exps[selected_exps], X_demo], axis=1)

    feature_sets = {
        "Exposure+Demo": X_expo_demo,
        "Demo_Only": X_demo
    }

    model_cfgs = get_model_configs(cfg.random_state)
    best_params = tune_hyperparams_once(
        X_expo_demo, y, model_cfgs,
        test_size=cfg.test_size_grid,
        random_state=cfg.random_state
    )

    all_rows = []
    for feat_name, X_feat in feature_sets.items():
        for model_name, mcfg in model_cfgs.items():
            model_ctor = mcfg["estimator"].__class__
            cv_df = repeated_stratified_cv(
                X_feat, y,
                model_name=model_name,
                model_ctor=model_ctor,
                best_params=best_params[model_name],
                n_iterations=cfg.n_iterations,
                n_folds=cfg.n_folds
            )
            cv_df["Disease"] = disease
            cv_df["Feature_Set"] = feat_name
            all_rows.append(cv_df)

    return pd.concat(all_rows, ignore_index=True)


def plot_auc_box(final_df: pd.DataFrame, plot_model: str, out_path: Path) -> None:
    d = final_df[final_df["Model"] == plot_model].copy()

    plt.figure(figsize=(18, 6))
    sns.boxplot(
        x="Disease",
        y="AUC",
        hue="Feature_Set",
        data=d,
        palette="Set2"
    )
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0.9, ls="--", color="gray", linewidth=1)
    plt.title(f"Prediction AUC for different diseases ({plot_model})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run repeated CV benchmarks for multiple diseases.")
    p.add_argument("--data", type=str, default=str(PROJECT_ROOT / "imputed_data.csv"),
                   help="Path to imputed_data.csv (or a sample csv).")
    p.add_argument("--exposure-map", type=str, default=str(PROJECT_ROOT / "configs" / "exposures_by_disease.csv"),
                   help="Path to exposures_by_disease.csv.")
    p.add_argument("--out", type=str, default=str(PROJECT_ROOT / "outputs"),
                   help="Output directory.")
    p.add_argument("--n-features", type=int, default=20, help="Top exposure features selected by SHAP.")
    p.add_argument("--iters", type=int, default=10, help="Repeated CV iterations.")
    p.add_argument("--folds", type=int, default=5, help="CV folds per iteration.")
    p.add_argument("--plot-model", type=str, default="LightGBM", help="Model name to plot AUC boxplot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = RunConfig(
        data_path=Path(args.data),
        exposure_map_path=Path(args.exposure_map),
        out_dir=Path(args.out),
        n_features_select=args.n_features,
        n_iterations=args.iters,
        n_folds=args.folds,
        plot_model=args.plot_model
    )

    ensure_dir(cfg.out_dir)

    df = pd.read_csv(cfg.data_path)
    exposure_map = load_exposure_mapping(cfg.exposure_map_path)

    diseases = [
        "heart_attack", "angina", "hypertension",
        "emphysema", "COPD", "chronic_bronchitis", "asthma",
        "diabetes", "attention_disorder", "osteoporosis", "arthritis"
    ]

    all_results = []
    for disease in diseases:
        print(f"\n=== Processing: {disease} ===")
        res = run_for_one_disease(df, disease, exposure_map, cfg)
        all_results.append(res)
        res.to_csv(cfg.out_dir / f"metrics_{disease}.csv", index=False)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(cfg.out_dir / "metrics_all_diseases.csv", index=False)

    summary = (
        final_df
        .groupby(["Disease", "Feature_Set", "Model"])[["AUC", "Accuracy", "Precision", "Recall", "F1"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(cfg.out_dir / "summary_mean_metrics.csv", index=False)

    print("\n=== Mean metrics ===")
    print(summary)

    plot_path = cfg.out_dir / f"auc_boxplot_{cfg.plot_model}.png"
    plot_auc_box(final_df, cfg.plot_model, plot_path)
    print(f"\nSaved plot: {plot_path}")


if __name__ == "__main__":
    main()
