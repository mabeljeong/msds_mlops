#!/usr/bin/env python3
"""Train, audit, and register the RentIQ XGBoost rent predictor (v2).

This script runs a small experiment suite under one MLflow parent run:
  - 3 target variants (raw_positive, winsorized_1_99, scope_2000_8000)
  - 5 hand-picked XGBoost hyperparameter configs
  - 1 ZORI-only baseline per variant
Then it picks the best (variant, hp) by CV MAE, retrains a tri-model bundle
(point + q10 + q90), and registers the final artifact for the API.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audit import run_audit  # noqa: E402
from app.rent_predictor import (  # noqa: E402
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RentPredictor,
    RentPredictorPyfunc,
    build_training_pipeline,
)

FEATURE_PATH = ROOT / "data" / "features" / "listings_features.csv"

CV_SPLITS = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class HPConfig:
    name: str
    max_depth: int
    min_child_weight: int
    n_estimators: int


HP_CONFIGS: list[HPConfig] = [
    HPConfig("depth3_mcw3_n220", max_depth=3, min_child_weight=3, n_estimators=220),
    HPConfig("depth3_mcw1_n300", max_depth=3, min_child_weight=1, n_estimators=300),
    HPConfig("depth4_mcw3_n200", max_depth=4, min_child_weight=3, n_estimators=200),
    HPConfig("depth4_mcw5_n150", max_depth=4, min_child_weight=5, n_estimators=150),
    HPConfig("depth5_mcw5_n150", max_depth=5, min_child_weight=5, n_estimators=150),
]


VariantFn = Callable[[pd.DataFrame], pd.DataFrame]


def _variant_raw_positive(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["rent_usd"] > 0].copy()


def _variant_winsorized(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["rent_usd"] > 0].copy()
    p1, p99 = out["rent_usd"].quantile([0.01, 0.99])
    out["rent_usd"] = out["rent_usd"].clip(lower=p1, upper=p99)
    return out


def _variant_scope_band(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df["rent_usd"] >= 2_000) & (df["rent_usd"] <= 8_000)].copy()


VARIANTS: dict[str, VariantFn] = {
    "raw_positive": _variant_raw_positive,
    "winsorized_1_99": _variant_winsorized,
    "scope_2000_8000": _variant_scope_band,
}


def _load_features() -> pd.DataFrame:
    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(f"Missing feature table: {FEATURE_PATH}")
    df = pd.read_csv(FEATURE_PATH, low_memory=False)
    df["rent_usd"] = pd.to_numeric(df["rent_usd"], errors="coerce")
    df = df[df["rent_usd"].notna()].copy()
    return df


def _xy_from_variant(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    X["zip_code"] = X["zip_code"].fillna("UNK").astype(str)
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    y_log = np.log1p(df["rent_usd"].astype(float))
    return X.reset_index(drop=True), y_log.reset_index(drop=True)


def _summarize(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    arr = pd.DataFrame(fold_metrics)
    summary: dict[str, float] = {}
    for metric in ("mae", "mape", "r2"):
        summary[f"cv_{metric}_mean"] = float(arr[metric].mean())
        summary[f"cv_{metric}_std"] = float(arr[metric].std(ddof=0))
    n = len(arr)
    if n > 1:
        # 95% CI half-width using t≈1.96 approximation; acceptable at small N for reporting.
        summary["cv_mae_ci95_halfwidth"] = float(1.96 * arr["mae"].std(ddof=1) / math.sqrt(n))
    else:
        summary["cv_mae_ci95_halfwidth"] = 0.0
    summary["cv_mae_min"] = float(arr["mae"].min())
    summary["cv_mae_max"] = float(arr["mae"].max())
    return summary


def _feature_importances_df(pipeline) -> pd.DataFrame:
    pre = pipeline.named_steps["preprocessor"]
    reg = pipeline.named_steps["regressor"]
    names = pre.get_feature_names_out()
    gains = reg.feature_importances_
    out = pd.DataFrame({"feature": names, "importance_gain": gains})
    return out.sort_values("importance_gain", ascending=False).reset_index(drop=True)


def _xgb_cv(
    X: pd.DataFrame,
    y_log: pd.Series,
    hp: HPConfig,
    objective: str = "reg:squarederror",
    quantile_alpha: float | None = None,
) -> tuple[list[dict[str, float]], list[pd.DataFrame], np.ndarray]:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: list[dict[str, float]] = []
    fold_importances: list[pd.DataFrame] = []
    oof_pred = np.zeros(len(X), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[valid_idx]
        pipe = build_training_pipeline(
            random_state=RANDOM_STATE + fold_idx,
            n_estimators=hp.n_estimators,
            max_depth=hp.max_depth,
            min_child_weight=hp.min_child_weight,
            objective=objective,
            quantile_alpha=quantile_alpha,
        )
        pipe.fit(X_tr, y_tr)
        pred_log = pipe.predict(X_val)
        oof_pred[valid_idx] = pred_log
        y_pred = np.expm1(pred_log)
        y_true = np.expm1(y_val)
        fold_metrics.append(
            {
                "fold": float(fold_idx),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
        )
        fold_importances.append(_feature_importances_df(pipe))
    return fold_metrics, fold_importances, oof_pred


def _zori_baseline_cv(df_variant: pd.DataFrame) -> list[dict[str, float]]:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    df = df_variant.reset_index(drop=True)
    y = pd.to_numeric(df["rent_usd"], errors="coerce").astype(float)
    zori = pd.to_numeric(df["zori_baseline"], errors="coerce").astype(float)
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(df), start=1):
        train_median = float(np.nanmedian(y.iloc[train_idx]))
        pred = zori.iloc[valid_idx].fillna(train_median).to_numpy()
        truth = y.iloc[valid_idx].to_numpy()
        fold_metrics.append(
            {
                "fold": float(fold_idx),
                "mae": float(mean_absolute_error(truth, pred)),
                "mape": float(mean_absolute_percentage_error(truth, pred)),
                "r2": float(r2_score(truth, pred)),
            }
        )
    return fold_metrics


def _log_fold_metrics(fold_metrics: list[dict[str, float]]) -> None:
    for m in fold_metrics:
        i = int(m["fold"])
        mlflow.log_metric(f"fold_{i}_mae", m["mae"])
        mlflow.log_metric(f"fold_{i}_mape", m["mape"])
        mlflow.log_metric(f"fold_{i}_r2", m["r2"])


def _log_artifact_dict(payload: dict | list, name: str, tmpdir: Path) -> None:
    tmpdir.mkdir(parents=True, exist_ok=True)
    p = tmpdir / name
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    mlflow.log_artifact(str(p), artifact_path="reports")


def _log_artifact_csv(df: pd.DataFrame, name: str, tmpdir: Path) -> None:
    tmpdir.mkdir(parents=True, exist_ok=True)
    p = tmpdir / name
    df.to_csv(p, index=False)
    mlflow.log_artifact(str(p), artifact_path="reports")


@dataclass
class ChildRunRecord:
    variant: str
    model_type: str
    hp_name: str | None
    cv_mae_mean: float
    cv_mae_std: float
    cv_mae_ci95_halfwidth: float
    cv_mape_mean: float
    cv_r2_mean: float
    leakage_warning: bool
    n_rows: int


def _train_baseline_child(
    df_variant: pd.DataFrame, variant_name: str, tmpdir: Path
) -> ChildRunRecord:
    fold_metrics = _zori_baseline_cv(df_variant)
    summary = _summarize(fold_metrics)
    with mlflow.start_run(run_name=f"baseline_zori__{variant_name}", nested=True) as run:
        mlflow.set_tags({"model_type": "baseline_zori", "variant": variant_name})
        mlflow.log_params(
            {
                "model_type": "baseline_zori",
                "variant": variant_name,
                "n_rows": int(len(df_variant)),
                "predictor": "zori_baseline_with_train_median_fallback",
                "n_splits": CV_SPLITS,
            }
        )
        for k, v in summary.items():
            mlflow.log_metric(k, v)
        _log_fold_metrics(fold_metrics)
        _log_artifact_dict(fold_metrics, "cv_fold_metrics.json", tmpdir / run.info.run_id)
    return ChildRunRecord(
        variant=variant_name,
        model_type="baseline_zori",
        hp_name=None,
        cv_mae_mean=summary["cv_mae_mean"],
        cv_mae_std=summary["cv_mae_std"],
        cv_mae_ci95_halfwidth=summary["cv_mae_ci95_halfwidth"],
        cv_mape_mean=summary["cv_mape_mean"],
        cv_r2_mean=summary["cv_r2_mean"],
        leakage_warning=False,
        n_rows=len(df_variant),
    )


def _train_xgb_point_child(
    X: pd.DataFrame,
    y_log: pd.Series,
    hp: HPConfig,
    variant_name: str,
    tmpdir: Path,
) -> ChildRunRecord:
    fold_metrics, fold_imps, _ = _xgb_cv(X, y_log, hp)
    summary = _summarize(fold_metrics)
    final_pipe = build_training_pipeline(
        random_state=RANDOM_STATE,
        n_estimators=hp.n_estimators,
        max_depth=hp.max_depth,
        min_child_weight=hp.min_child_weight,
    )
    final_pipe.fit(X, y_log)
    importance_df = _feature_importances_df(final_pipe)

    audit = run_audit(
        X=X,
        y=y_log,
        feature_names=importance_df["feature"].tolist(),
        importances=importance_df["importance_gain"].tolist(),
        fold_importance_frames=fold_imps,
    )

    with mlflow.start_run(run_name=f"xgb_point__{variant_name}__{hp.name}", nested=True) as run:
        mlflow.set_tags(
            {
                "model_type": "xgb_point",
                "variant": variant_name,
                "hp_config": hp.name,
                "leakage_warning": str(audit.has_warning).lower(),
            }
        )
        mlflow.log_params(
            {
                "model_type": "xgb_point",
                "variant": variant_name,
                "hp_config": hp.name,
                "max_depth": hp.max_depth,
                "min_child_weight": hp.min_child_weight,
                "n_estimators": hp.n_estimators,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
                "objective": "reg:squarederror",
                "n_splits": CV_SPLITS,
                "n_rows": int(X.shape[0]),
                "n_features": int(X.shape[1]),
            }
        )
        for k, v in summary.items():
            mlflow.log_metric(k, v)
        _log_fold_metrics(fold_metrics)

        run_dir = tmpdir / run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _log_artifact_dict(fold_metrics, "cv_fold_metrics.json", run_dir)
        _log_artifact_csv(importance_df, "feature_importances.csv", run_dir)
        _log_artifact_dict(audit.to_dict(), "leakage_audit.json", run_dir)
        if audit.warnings:
            mlflow.set_tag("audit_warnings", ";".join(audit.warnings))

    return ChildRunRecord(
        variant=variant_name,
        model_type="xgb_point",
        hp_name=hp.name,
        cv_mae_mean=summary["cv_mae_mean"],
        cv_mae_std=summary["cv_mae_std"],
        cv_mae_ci95_halfwidth=summary["cv_mae_ci95_halfwidth"],
        cv_mape_mean=summary["cv_mape_mean"],
        cv_r2_mean=summary["cv_r2_mean"],
        leakage_warning=audit.has_warning,
        n_rows=int(X.shape[0]),
    )


def _coverage_from_cv(
    X: pd.DataFrame,
    y_log: pd.Series,
    hp: HPConfig,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    coverage_per_fold: list[dict[str, float]] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[valid_idx]
        q10 = build_training_pipeline(
            random_state=RANDOM_STATE + fold_idx,
            n_estimators=hp.n_estimators,
            max_depth=hp.max_depth,
            min_child_weight=hp.min_child_weight,
            objective="reg:quantileerror",
            quantile_alpha=0.1,
        )
        q90 = build_training_pipeline(
            random_state=RANDOM_STATE + fold_idx,
            n_estimators=hp.n_estimators,
            max_depth=hp.max_depth,
            min_child_weight=hp.min_child_weight,
            objective="reg:quantileerror",
            quantile_alpha=0.9,
        )
        q10.fit(X_tr, y_tr)
        q90.fit(X_tr, y_tr)
        pred_lo = np.expm1(q10.predict(X_val))
        pred_hi = np.expm1(q90.predict(X_val))
        truth = np.expm1(y_val)
        inside = ((truth >= pred_lo) & (truth <= pred_hi)).mean()
        width = float(np.mean(np.maximum(pred_hi - pred_lo, 0.0)))
        coverage_per_fold.append(
            {
                "fold": float(fold_idx),
                "coverage_p10_p90": float(inside),
                "mean_interval_width_usd": width,
            }
        )
    arr = pd.DataFrame(coverage_per_fold)
    summary = {
        "interval_coverage_mean": float(arr["coverage_p10_p90"].mean()),
        "interval_coverage_std": float(arr["coverage_p10_p90"].std(ddof=0)),
        "interval_width_mean_usd": float(arr["mean_interval_width_usd"].mean()),
    }
    return summary, coverage_per_fold


def _select_best(records: list[ChildRunRecord]) -> ChildRunRecord:
    point_runs = [r for r in records if r.model_type == "xgb_point"]
    if not point_runs:
        raise RuntimeError("No xgb_point runs to select from.")
    point_runs.sort(key=lambda r: (r.cv_mae_mean, r.cv_mae_std))
    return point_runs[0]


def _baseline_for_variant(records: list[ChildRunRecord], variant: str) -> ChildRunRecord | None:
    for r in records:
        if r.variant == variant and r.model_type == "baseline_zori":
            return r
    return None


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str((ROOT / "mlruns").as_uri()))
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "RentIQ")
    registered_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "RentIQRentPredictor")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df_full = _load_features()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with mlflow.start_run(run_name="rentiq_v2_selection") as parent:
            mlflow.set_tags(
                {
                    "experiment_phase": "selection_and_register",
                    "feature_table": str(FEATURE_PATH.relative_to(ROOT)),
                }
            )
            mlflow.log_params(
                {
                    "feature_table_rows": int(len(df_full)),
                    "feature_columns": ",".join(FEATURE_COLUMNS),
                    "n_splits": CV_SPLITS,
                    "random_state": RANDOM_STATE,
                    "variants": ",".join(VARIANTS.keys()),
                    "hp_configs": ",".join(h.name for h in HP_CONFIGS),
                }
            )

            records: list[ChildRunRecord] = []
            variant_xy: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}

            for variant_name, variant_fn in VARIANTS.items():
                df_v = variant_fn(df_full)
                if len(df_v) < 50:
                    continue
                X_v, y_v = _xy_from_variant(df_v)
                variant_xy[variant_name] = (X_v, y_v, df_v.reset_index(drop=True))

                records.append(_train_baseline_child(df_v, variant_name, tmpdir))
                for hp in HP_CONFIGS:
                    records.append(_train_xgb_point_child(X_v, y_v, hp, variant_name, tmpdir))

            best = _select_best(records)
            best_baseline = _baseline_for_variant(records, best.variant)
            mae_lift_vs_zori = (
                (best_baseline.cv_mae_mean - best.cv_mae_mean) / best_baseline.cv_mae_mean
                if best_baseline and best_baseline.cv_mae_mean > 0
                else None
            )

            summary_rows = [
                {
                    "variant": r.variant,
                    "model_type": r.model_type,
                    "hp_config": r.hp_name or "",
                    "cv_mae_mean": r.cv_mae_mean,
                    "cv_mae_std": r.cv_mae_std,
                    "cv_mae_ci95_halfwidth": r.cv_mae_ci95_halfwidth,
                    "cv_mape_mean": r.cv_mape_mean,
                    "cv_r2_mean": r.cv_r2_mean,
                    "leakage_warning": r.leakage_warning,
                    "n_rows": r.n_rows,
                }
                for r in records
            ]
            summary_df = pd.DataFrame(summary_rows).sort_values(
                ["cv_mae_mean", "cv_mae_std"]
            )
            _log_artifact_csv(summary_df, "model_selection_summary.csv", tmpdir)

            X_best, y_best, _ = variant_xy[best.variant]
            best_hp = next(h for h in HP_CONFIGS if h.name == best.hp_name)

            point_pipe = build_training_pipeline(
                random_state=RANDOM_STATE,
                n_estimators=best_hp.n_estimators,
                max_depth=best_hp.max_depth,
                min_child_weight=best_hp.min_child_weight,
            )
            point_pipe.fit(X_best, y_best)
            q10_pipe = build_training_pipeline(
                random_state=RANDOM_STATE,
                n_estimators=best_hp.n_estimators,
                max_depth=best_hp.max_depth,
                min_child_weight=best_hp.min_child_weight,
                objective="reg:quantileerror",
                quantile_alpha=0.1,
            )
            q10_pipe.fit(X_best, y_best)
            q90_pipe = build_training_pipeline(
                random_state=RANDOM_STATE,
                n_estimators=best_hp.n_estimators,
                max_depth=best_hp.max_depth,
                min_child_weight=best_hp.min_child_weight,
                objective="reg:quantileerror",
                quantile_alpha=0.9,
            )
            q90_pipe.fit(X_best, y_best)

            coverage_summary, coverage_per_fold = _coverage_from_cv(X_best, y_best, best_hp)

            predictor = RentPredictor(
                model=point_pipe,
                feature_columns=FEATURE_COLUMNS,
                q10_model=q10_pipe,
                q90_model=q90_pipe,
                metadata={
                    "variant": best.variant,
                    "hp_config": best.hp_name,
                    "registered_model_name": registered_name,
                    "trained_rows": int(X_best.shape[0]),
                },
            )

            with mlflow.start_run(run_name="rentiq_v2_final_bundle", nested=True) as final_run:
                mlflow.set_tags(
                    {
                        "model_type": "tri_model_bundle",
                        "selected_variant": best.variant,
                        "selected_hp": best.hp_name,
                    }
                )
                mlflow.log_params(
                    {
                        "selected_variant": best.variant,
                        "selected_hp": best.hp_name,
                        "max_depth": best_hp.max_depth,
                        "min_child_weight": best_hp.min_child_weight,
                        "n_estimators": best_hp.n_estimators,
                        "n_rows": int(X_best.shape[0]),
                    }
                )
                mlflow.log_metric("cv_mae_mean", best.cv_mae_mean)
                mlflow.log_metric("cv_mae_std", best.cv_mae_std)
                mlflow.log_metric("cv_mae_ci95_halfwidth", best.cv_mae_ci95_halfwidth)
                mlflow.log_metric("cv_mape_mean", best.cv_mape_mean)
                mlflow.log_metric("cv_r2_mean", best.cv_r2_mean)
                for k, v in coverage_summary.items():
                    mlflow.log_metric(k, v)
                if mae_lift_vs_zori is not None:
                    mlflow.log_metric("mae_lift_vs_zori", float(mae_lift_vs_zori))
                    mlflow.set_tag(
                        "value_add_lt_10pct",
                        str(mae_lift_vs_zori < 0.10).lower(),
                    )

                run_dir = tmpdir / final_run.info.run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                _log_artifact_dict(coverage_per_fold, "interval_coverage_per_fold.json", run_dir)
                _log_artifact_dict(coverage_summary, "interval_coverage_summary.json", run_dir)
                _log_artifact_csv(
                    _feature_importances_df(point_pipe),
                    "feature_importances_final.csv",
                    run_dir,
                )

                bundle_path = run_dir / "rent_predictor.joblib"
                joblib.dump(predictor, bundle_path)

                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=RentPredictorPyfunc(),
                    artifacts={"predictor_bundle": str(bundle_path)},
                    registered_model_name=registered_name,
                )

                final_run_id = final_run.info.run_id

            mlflow.log_metric("final_cv_mae_mean", best.cv_mae_mean)
            if mae_lift_vs_zori is not None:
                mlflow.log_metric("final_mae_lift_vs_zori", float(mae_lift_vs_zori))
                mlflow.set_tag(
                    "value_add_lt_10pct",
                    str(mae_lift_vs_zori < 0.10).lower(),
                )

    print()
    print("=" * 60)
    print(f"Best config: variant={best.variant}, hp={best.hp_name}")
    print(
        f"CV MAE mean: {best.cv_mae_mean:.2f} "
        f"(±{best.cv_mae_ci95_halfwidth:.2f} 95% CI; std {best.cv_mae_std:.2f})"
    )
    print(f"CV MAPE mean: {best.cv_mape_mean:.4f}")
    print(f"CV R2 mean:   {best.cv_r2_mean:.4f}")
    if mae_lift_vs_zori is not None:
        print(f"MAE lift vs ZORI baseline: {mae_lift_vs_zori * 100:.2f}%")
    print(
        "Interval coverage P(actual ∈ [p10,p90]): "
        f"{coverage_summary['interval_coverage_mean'] * 100:.1f}% "
        f"(width≈${coverage_summary['interval_width_mean_usd']:.0f})"
    )
    print()
    print("Set these for API loading:")
    print(f'export MLFLOW_TRACKING_URI="{tracking_uri}"')
    print(f'export MLFLOW_MODEL_URI="runs:/{final_run_id}/model"')
    print(f'export MLFLOW_MODEL_VERSION="{final_run_id[:8]}"')
    print(f'export MLFLOW_REGISTERED_MODEL_NAME="{registered_name}"')


if __name__ == "__main__":
    main()
