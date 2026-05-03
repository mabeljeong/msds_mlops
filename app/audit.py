"""Leakage and degeneracy audit helpers used during training.

These utilities run before model selection so each MLflow run can ship
artifacts and warning tags that the parent run uses to filter risky
configurations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CORR_LEAKAGE_THRESHOLD = 0.95
DOMINANT_IMPORTANCE_THRESHOLD = 0.50


@dataclass
class AuditResult:
    target_correlations: list[dict[str, float | str]] = field(default_factory=list)
    near_deterministic_features: list[dict[str, float | str]] = field(default_factory=list)
    importance_share_top: list[dict[str, float | str]] = field(default_factory=list)
    fold_importance_stability: list[dict[str, float | str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_correlations": self.target_correlations,
            "near_deterministic_features": self.near_deterministic_features,
            "importance_share_top": self.importance_share_top,
            "fold_importance_stability": self.fold_importance_stability,
            "warnings": self.warnings,
        }

    @property
    def has_warning(self) -> bool:
        return bool(self.warnings)


def _corr_with_target(X: pd.DataFrame, y: pd.Series, method: str) -> pd.Series:
    numeric = X.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.Series(dtype=float)
    df = numeric.copy()
    df["__target__"] = pd.to_numeric(y, errors="coerce")
    return (
        df.corr(method=method, numeric_only=True)["__target__"]
        .drop(labels=["__target__"], errors="ignore")
        .dropna()
    )


def target_correlations(X: pd.DataFrame, y: pd.Series) -> list[dict[str, float | str]]:
    pearson = _corr_with_target(X, y, method="pearson")
    spearman = _corr_with_target(X, y, method="spearman")
    feats = sorted(set(pearson.index) | set(spearman.index))
    rows: list[dict[str, float | str]] = []
    for f in feats:
        rows.append(
            {
                "feature": str(f),
                "pearson": float(pearson.get(f, np.nan)),
                "spearman": float(spearman.get(f, np.nan)),
            }
        )
    rows.sort(key=lambda r: abs(float(r["pearson"]) if not np.isnan(float(r["pearson"])) else 0.0), reverse=True)
    return rows


def near_deterministic(X: pd.DataFrame, y: pd.Series, threshold: float = CORR_LEAKAGE_THRESHOLD) -> list[dict[str, float | str]]:
    pearson = _corr_with_target(X, y, method="pearson")
    flagged: list[dict[str, float | str]] = []
    for feat, value in pearson.items():
        if pd.notna(value) and abs(float(value)) >= threshold:
            flagged.append({"feature": str(feat), "pearson": float(value)})
    flagged.sort(key=lambda r: abs(float(r["pearson"])), reverse=True)
    return flagged


def importance_share(
    feature_names: Iterable[str],
    importances: Iterable[float],
    top_k: int = 10,
) -> list[dict[str, float | str]]:
    names = list(feature_names)
    gains = np.asarray(list(importances), dtype=float)
    total = float(gains.sum()) if gains.size else 0.0
    rows: list[dict[str, float | str]] = []
    for n, g in zip(names, gains):
        rows.append(
            {
                "feature": str(n),
                "importance_gain": float(g),
                "share_of_total": float(g / total) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda r: float(r["importance_gain"]), reverse=True)
    return rows[:top_k]


def fold_importance_stability(
    fold_importance_frames: list[pd.DataFrame],
    top_k: int = 10,
) -> list[dict[str, float | str]]:
    """Each frame is a per-fold (feature, importance_gain) DataFrame."""
    if not fold_importance_frames:
        return []
    merged = None
    for i, frame in enumerate(fold_importance_frames):
        renamed = frame.rename(columns={"importance_gain": f"fold_{i + 1}"}).copy()
        renamed = renamed[["feature", f"fold_{i + 1}"]]
        merged = renamed if merged is None else merged.merge(renamed, on="feature", how="outer")
    if merged is None:
        return []
    fold_cols = [c for c in merged.columns if c.startswith("fold_")]
    if not fold_cols:
        return []
    merged = merged.fillna(0.0)
    merged["mean_importance"] = merged[fold_cols].mean(axis=1)
    merged["std_importance"] = merged[fold_cols].std(axis=1, ddof=0)
    merged["coefficient_of_variation"] = (
        merged["std_importance"] / merged["mean_importance"].replace({0.0: np.nan})
    )
    merged = merged.sort_values("mean_importance", ascending=False).head(top_k)

    rows: list[dict[str, float | str]] = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "feature": str(r["feature"]),
                "mean_importance": float(r["mean_importance"]),
                "std_importance": float(r["std_importance"]),
                "coefficient_of_variation": (
                    float(r["coefficient_of_variation"])
                    if pd.notna(r["coefficient_of_variation"])
                    else None
                ),
            }
        )
    return rows


def run_audit(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Iterable[str] | None = None,
    importances: Iterable[float] | None = None,
    fold_importance_frames: list[pd.DataFrame] | None = None,
) -> AuditResult:
    """Run all audit checks and assemble a result with warnings."""
    result = AuditResult()
    result.target_correlations = target_correlations(X, y)
    result.near_deterministic_features = near_deterministic(X, y)
    if feature_names is not None and importances is not None:
        result.importance_share_top = importance_share(feature_names, importances)
    if fold_importance_frames:
        result.fold_importance_stability = fold_importance_stability(fold_importance_frames)

    if result.near_deterministic_features:
        result.warnings.append(
            f"near_deterministic_features_detected:{len(result.near_deterministic_features)}"
        )
    if result.importance_share_top:
        top = result.importance_share_top[0]
        if float(top["share_of_total"]) >= DOMINANT_IMPORTANCE_THRESHOLD:
            result.warnings.append(
                f"dominant_feature:{top['feature']}={float(top['share_of_total']):.3f}"
            )
    if result.fold_importance_stability:
        unstable = [
            r for r in result.fold_importance_stability
            if r.get("coefficient_of_variation") is not None
            and float(r["coefficient_of_variation"]) > 1.0
        ]
        if unstable:
            result.warnings.append(f"unstable_top_importances:{len(unstable)}")

    return result
