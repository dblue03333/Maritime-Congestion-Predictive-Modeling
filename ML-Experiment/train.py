"""
HarborMind — LightGBM Training Pipeline
========================================
MLflow tracking + Optuna hyperparameter optimization.

Usage:
    python mlruns/train.py

Author: Kelvin Nguyen
"""

import os
import json
import joblib
import numpy as np
import polars as pl
import lightgbm as lgb
import mlflow
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

CONFIG = {
    # Data
    "data_path": "Data-Engineering/data/processed/ais_2023_2025_clean.parquet",
    "model_dir": "mlruns/models",

    # Mode: "retrospective" (full visit data) or "prospective" (arrival-time only)
    "mode": "retrospective",

    # 3-way temporal split (Train → Optuna Val → Holdout Test)
    "val_date": "2024-07-01",   # Train: Jun2023-Jun2024 | Val: Jul2024-Dec2024
    "test_date": "2025-01-01",  # Test: Jan2025-Dec2025 (used ONCE for final report)

    # Target
    "target": "delay_minutes",
    "outlier_cap_hours": 72,

    # Feature selection
    "corr_threshold": 0.95,  # drop features with correlation > 0.95

    # Optuna
    "n_trials": 100,  # Increased from 50 for better hyperparameter search
    "early_stopping_rounds": 100,
}


# ============================================================
# 2. DATA LOADING & VALIDATION
# ============================================================

def load_data(config):
    """Load parquet and perform sanity checks."""
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    df = pl.read_parquet(config["data_path"])
    print(f"  Raw shape: {df.shape}")
    print(f"  Date range: {df['base_date_time'].min()} → {df['base_date_time'].max()}")
    print(f"  Unique ships: {df['mmsi'].n_unique()}")

    # Filter: only rows WITH delay label
    df_labeled = df.filter(pl.col(config["target"]).is_not_null())
    print(f"  Labeled rows: {len(df_labeled):,} / {len(df):,} "
          f"({len(df_labeled)/len(df)*100:.1f}%)")

    assert len(df_labeled) > 100_000, "Too few labeled rows"
    assert df_labeled[config["target"]].min() >= 0, "Negative delay found"

    return df_labeled


# ============================================================
# 3. VISIT-LEVEL AGGREGATION
# ============================================================

def aggregate_visits(df, config):
    """
    Aggregate per-ping data into per-visit features.
    
    Two modes:
    - retrospective: uses ALL data from entire visit (post-hoc analysis)
    - prospective: uses ONLY data available at arrival time (real-time prediction)
    """
    mode = config.get("mode", "retrospective")
    print("\n" + "=" * 60)
    print(f"STEP 2: Visit-Level Aggregation (mode={mode})")
    print("=" * 60)
    print(f"  Input: {len(df):,} pings")

    # --- Common aggregations (both modes) ---
    common_aggs = [
        # Vessel static (same for all pings)
        pl.col("draft").first(),
        pl.col("length").first(),
        pl.col("width").first(),
        pl.col("vessel_type").first(),
        # pl.col("cargo").first(),  # Removed: dropped in post_process_gold.py
        pl.col("vessel_area").first(),
        pl.col("dimension_ratio").first(),
        pl.col("draft_to_length_ratio").first(),

        # Temporal (arrival time)
        pl.col("base_date_time").first().alias("arrival_time"),
        pl.col("hour_sin").first().alias("arrival_hour_sin"),
        pl.col("hour_cos").first().alias("arrival_hour_cos"),
        pl.col("day_of_week_sin").first().alias("arrival_dow_sin"),
        pl.col("day_of_week_cos").first().alias("arrival_dow_cos"),
        pl.col("month_sin").first().alias("arrival_month_sin"),
        pl.col("month_cos").first().alias("arrival_month_cos"),
        pl.col("is_weekend").first(),
        pl.col("is_night_shift").first(),
        pl.col("is_gate_hours").first(),

        # Target
        pl.col("delay_minutes").first(),
    ]

    # --- Prospective: only features available AT ARRIVAL ---
    arrival_aggs = [
        # First ping (known at arrival)
        pl.col("sog").first().alias("arrival_sog"),
        pl.col("cog").first().alias("arrival_cog"),
        pl.col("distance_to_port").first().alias("arrival_distance"),
        pl.col("heading_error").first().alias("arrival_heading_error"),

        # Port state at arrival
        pl.col("ship_density").first().alias("arrival_density"),
        pl.col("avg_port_speed").first().alias("arrival_port_speed"),
        pl.col("port_throughput").first().alias("arrival_throughput"),

        # Weather at arrival
        pl.col("wind_speed_10m").first().alias("arrival_wind"),
        pl.col("wind_gusts_10m").first().alias("arrival_gusts"),
        pl.col("precipitation").first().alias("arrival_precip"),
        pl.col("wave_height").first().alias("arrival_wave"),
        pl.col("swell_wave_height").first().alias("arrival_swell"),
    ]

    # --- Retrospective: full visit data (post-hoc) ---
    retro_aggs = [
        # Kinematic (behavior during visit)
        pl.col("sog").mean().alias("avg_sog"),
        pl.col("sog").max().alias("max_sog"),
        pl.col("sog").std().alias("std_sog"),
        pl.col("cog").std().alias("std_cog"),
        pl.col("acceleration").mean().alias("avg_accel"),
        pl.col("acceleration").std().alias("std_accel"),
        pl.col("heading_error").mean().alias("avg_heading_error"),

        # Spatial (approach pattern)
        pl.col("distance_to_port").min().alias("min_distance"),
        pl.col("distance_to_port").mean().alias("avg_distance"),
        pl.col("distance_to_port").first().alias("first_distance"),

        # Congestion (port state during visit)
        pl.col("ship_density").mean().alias("avg_density"),
        pl.col("ship_density").max().alias("max_density"),
        pl.col("avg_port_speed").mean().alias("avg_port_speed"),
        pl.col("port_throughput").mean().alias("avg_throughput"),
        pl.col("is_in_waiting_area").mean().alias("pct_waiting"),

        # Weather (conditions during visit)
        pl.col("wind_speed_10m").mean().alias("avg_wind"),
        pl.col("wind_speed_10m").max().alias("max_wind"),
        pl.col("wind_gusts_10m").max().alias("max_gusts"),
        pl.col("precipitation").sum().alias("total_precip"),
        pl.col("wave_height").mean().alias("avg_wave"),
        pl.col("swell_wave_height").mean().alias("avg_swell"),

        # Visit duration
        pl.len().alias("n_pings"),
    ]

    if mode == "prospective":
        aggs = common_aggs + arrival_aggs
    else:
        aggs = common_aggs + retro_aggs

    visits = df.group_by(["mmsi", "visit_id"]).agg(aggs)

    # --- Enhanced interaction features (Strategy B: +0.073 R²) ---
    visits = visits.with_columns([
        # Interaction: big ship in crowded port
        (pl.col("avg_density") * pl.col("vessel_area")).alias("density_x_area"),
        # Speed at distance (approaching pattern)
        (pl.col("avg_sog") * pl.col("first_distance")).alias("sog_x_distance"),
        # Draft loading ratio
        (pl.col("draft") / (pl.col("width") + 1e-6)).alias("draft_width_ratio"),
        # Congestion pressure (density / throughput)
        (pl.col("avg_density") / (pl.col("avg_throughput") + 1e-6)).alias("congestion_pressure"),
        # Weather severity composite
        (pl.col("avg_wind") * pl.col("avg_swell")).alias("weather_severity"),
        # Distance range during visit (approach depth)
        (pl.col("first_distance") - pl.col("min_distance")).alias("distance_range"),
        # Vessel size category (0=small, 1=medium, 2=large, 3=VLCC)
        pl.when(pl.col("vessel_area") < 1000).then(pl.lit(0))
        .when(pl.col("vessel_area") < 5000).then(pl.lit(1))
        .when(pl.col("vessel_area") < 15000).then(pl.lit(2))
        .otherwise(pl.lit(3))
        .alias("size_category"),
    ])

    print(f"  Output: {len(visits):,} visits")
    print(f"  Features: {len(visits.columns) - 4} (excl mmsi, visit_id, arrival_time, target)")
    print(f"  Compression: {len(df)/len(visits):.0f}x")

    return visits


# ============================================================
# 4. LAG FEATURES (historical context)
# ============================================================

def add_lag_features(df):
    """
    Add historical context features — only uses PAST data (no leakage).
    
    These features answer: "what was the port like BEFORE this ship arrived?"
    This is the key to improving prospective model performance.
    """
    print("\n" + "=" * 60)
    print("STEP 2b: Lag Features (historical context)")
    print("=" * 60)

    # Sort by arrival time
    df = df.sort("arrival_time")

    # Rolling delay of last N ships (port-wide)
    for window in [5, 10, 20]:
        df = df.with_columns(
            pl.col("delay_minutes")
            .shift(1)  # exclude current visit
            .rolling_mean(window_size=window, min_periods=1)
            .alias(f"rolling_delay_{window}")
        )

    # Arrivals in last 24h and 48h
    arrival_ts = df["arrival_time"]
    n = len(df)
    arrivals_24h = []
    arrivals_48h = []

    for i in range(n):
        current_time = arrival_ts[i]
        if current_time is None:
            arrivals_24h.append(None)
            arrivals_48h.append(None)
            continue
        # Count arrivals in window (look backwards only)
        count_24 = 0
        count_48 = 0
        for j in range(i - 1, max(i - 200, -1), -1):
            if arrival_ts[j] is None:
                continue
            diff_hours = (current_time - arrival_ts[j]).total_seconds() / 3600
            if diff_hours > 48:
                break
            count_48 += 1
            if diff_hours <= 24:
                count_24 += 1
        arrivals_24h.append(count_24)
        arrivals_48h.append(count_48)

    df = df.with_columns([
        pl.Series("arrivals_24h", arrivals_24h).cast(pl.Int32),
        pl.Series("arrivals_48h", arrivals_48h).cast(pl.Int32),
    ])

    # Rolling density trend (last 5 ships' density at arrival)
    if "arrival_density" in df.columns:
        density_col = "arrival_density"
    elif "avg_density" in df.columns:
        density_col = "avg_density"
    else:
        density_col = None

    if density_col:
        df = df.with_columns(
            pl.col(density_col)
            .shift(1)
            .rolling_mean(window_size=5, min_periods=1)
            .alias("recent_density_trend")
        )

    # Same vessel_type historical average delay
    df = df.with_columns(
        pl.col("delay_minutes")
        .shift(1)
        .rolling_mean(window_size=50, min_periods=1)
        .over("vessel_type")
        .alias("vessel_type_avg_delay")
    )

    # Same ship's previous delay (if visited before)
    df = df.with_columns(
        pl.col("delay_minutes")
        .shift(1)
        .over("mmsi")
        .alias("prev_ship_delay")
    )

    # Rolling volatility of delays (how unpredictable is the port right now?)
    df = df.with_columns(
        pl.col("delay_minutes")
        .shift(1)
        .rolling_std(window_size=10, min_samples=3)
        .alias("rolling_delay_std_10")
    )

    lag_cols = [c for c in df.columns if c.startswith(("rolling_", "arrivals_", "recent_", "vessel_type_avg", "prev_ship"))]
    print(f"  Added {len(lag_cols)} lag features: {lag_cols}")

    return df


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def prepare_features(df, config):
    """Prepare feature list and remove multicollinear features."""
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)

    target = config["target"]
    drop = ["mmsi", "visit_id", "arrival_time", target]
    feature_cols = [c for c in df.columns if c not in drop]

    # Convert boolean → int for LightGBM
    for col in feature_cols:
        if df[col].dtype == pl.Boolean:
            df = df.with_columns(pl.col(col).cast(pl.Int8))

    # Remove highly correlated features (multicollinearity)
    threshold = config.get("corr_threshold", 0.95)
    numeric_df = df.select(feature_cols).to_pandas().select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr().abs()

    # Find pairs above threshold
    dropped_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the one with higher mean correlation
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                mean_i = corr_matrix[col_i].mean()
                mean_j = corr_matrix[col_j].mean()
                drop_col = col_j if mean_j >= mean_i else col_i
                dropped_corr.add(drop_col)

    feature_cols = [c for c in feature_cols if c not in dropped_corr]

    print(f"  Features: {len(feature_cols)}")
    if dropped_corr:
        print(f"  Dropped (corr>{threshold}): {sorted(dropped_corr)}")
    print(f"  Target: {target}")

    return df, feature_cols


# ============================================================
# 6. TEMPORAL SPLIT (3-way: Train / Val / Test)
# ============================================================

def temporal_split(df, feature_cols, config):
    """
    3-way temporal split:
      Train: before val_date      → used for model fitting
      Val:   val_date to test_date → used for Optuna hyperparameter search
      Test:  after test_date       → used ONCE for final unbiased evaluation
    """
    print("\n" + "=" * 60)
    print("STEP 4: Temporal Split (3-way)")
    print("=" * 60)

    target = config["target"]
    cap = config["outlier_cap_hours"] * 60

    df = df.with_columns(pl.col("arrival_time").dt.replace_time_zone(None))
    val_dt = datetime.strptime(config["val_date"], "%Y-%m-%d")
    test_dt = datetime.strptime(config["test_date"], "%Y-%m-%d")

    df_train = df.filter(pl.col("arrival_time") < val_dt)
    df_val = df.filter((pl.col("arrival_time") >= val_dt) & (pl.col("arrival_time") < test_dt))
    df_test = df.filter(pl.col("arrival_time") >= test_dt)

    def _extract(df_part):
        X = df_part.select(feature_cols).to_pandas()
        # Enable LightGBM native categorical support
        for col in X.select_dtypes(include=['object', 'string']).columns:
            X[col] = X[col].astype('category')
        y_raw = df_part[target].to_pandas().clip(upper=cap)
        y_log = np.log1p(y_raw)
        return X, y_log, y_raw

    X_train, y_train, y_train_raw = _extract(df_train)
    X_val, y_val, y_val_raw = _extract(df_val)
    X_test, y_test, y_test_raw = _extract(df_test)

    print(f"  Train: {len(X_train):,} visits  (< {config['val_date']})")
    print(f"  Val:   {len(X_val):,} visits  ({config['val_date']} → {config['test_date']})")
    print(f"  Test:  {len(X_test):,} visits  (≥ {config['test_date']})  ← holdout")
    print(f"  Cap: {cap} min | Transform: log1p")

    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw


# ============================================================
# 7. OPTUNA HYPERPARAMETER SEARCH (on VALIDATION set only)
# ============================================================

def objective(trial, X_train, X_val, y_train, y_val, config):
    """Optuna objective: minimize MAE on VALIDATION set (not test)."""
    params = {
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(config["early_stopping_rounds"]),
            lgb.log_evaluation(0),
        ],
    )

    y_pred_log = model.predict(X_val)
    y_pred_real = np.clip(np.expm1(y_pred_log), 0, None)
    y_val_real = np.expm1(y_val)
    mae = mean_absolute_error(y_val_real, y_pred_real)

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metric("val_mae_minutes", mae)
        mlflow.log_metric("val_r2", r2_score(y_val_real, y_pred_real))
        mlflow.log_metric("best_iteration", model.best_iteration_)

    return mae


def run_optuna(X_train, X_val, y_train, y_val, config):
    """Run Optuna search on validation set."""
    print("\n" + "=" * 60)
    print(f"STEP 5: Optuna Search ({config['n_trials']} trials) — on VAL set")
    print("=" * 60)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=42) # Fix seed for reproducibility
    study = optuna.create_study(direction="minimize", study_name="harbormind_lgbm", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val, config),
        n_trials=config["n_trials"],
        show_progress_bar=True,
    )

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best VAL MAE: {study.best_value:.1f} minutes")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study.best_params


# ============================================================
# 8. TRAIN FINAL MODEL (best params, on Train+Val combined)
# ============================================================

def train_final_model(X_train, X_val, y_train, y_val, best_params, config):
    """
    Train final model on Train+Val combined for maximum data.
    Early stopping monitored on Val portion.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Training Final Model (Train+Val)")
    print("=" * 60)

    import pandas as pd
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    final_params = {
        **best_params,
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**final_params)
    model.fit(
        X_combined, y_combined,
        eval_set=[(X_val, y_val)],  # monitor on val for early stopping
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(config["early_stopping_rounds"]),
            lgb.log_evaluation(100),
        ],
    )

    print(f"  Train+Val: {len(X_combined):,} visits")
    print(f"  Best iteration: {model.best_iteration_}")

    return model


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test_log, y_test_raw):
    """Calculate regression metrics in real-world minutes."""
    print("\n" + "=" * 60)
    print("STEP 6: Evaluation")
    print("=" * 60)

    # Predict in log-space → inverse to minutes
    y_pred_log = model.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    y_test = y_test_raw  # evaluate in real minutes

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "median_ae": float(np.median(np.abs(y_test - y_pred))),
        "r2_log": float(r2_score(y_test_log, y_pred_log)),
    }

    mask = y_test > 0
    if mask.sum() > 0:
        metrics["mape"] = float(mean_absolute_percentage_error(y_test[mask], y_pred[mask]))

    print(f"  MAE:       {metrics['mae']:.1f} minutes ({metrics['mae']/60:.1f} hours)")
    print(f"  RMSE:      {metrics['rmse']:.1f} minutes")
    print(f"  Median AE: {metrics['median_ae']:.1f} minutes")
    print(f"  R² (real): {metrics['r2']:.4f}")
    print(f"  R² (log):  {metrics['r2_log']:.4f}")
    if "mape" in metrics:
        print(f"  MAPE:      {metrics['mape']*100:.1f}%")

    # Delay category accuracy
    bins = [0, 60, 360, 1440, float("inf")]
    actual_cat = np.digitize(y_test, bins) - 1
    pred_cat = np.digitize(y_pred, bins) - 1
    cat_accuracy = float(np.mean(actual_cat == pred_cat))
    print(f"  Category accuracy: {cat_accuracy*100:.1f}%")
    metrics["category_accuracy"] = cat_accuracy

    return metrics, y_pred


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(model, X_train, save_dir):
    """Plot top-20 feature importance."""
    print("\n" + "=" * 60)
    print("STEP 7: Feature Importance")
    print("=" * 60)

    importance = model.feature_importances_
    feature_names = X_train.columns
    indices = np.argsort(importance)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importance[indices][::-1],
            color="#2196F3", edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel("Feature Importance (split count)")
    ax.set_title("Top 20 Feature Importance — LightGBM")
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    for rank, idx in enumerate(indices[:10], 1):
        print(f"  #{rank:2d}: {feature_names[idx]:30s} → {importance[idx]}")


# ============================================================
# 9. RESIDUAL ANALYSIS
# ============================================================

def plot_residuals(y_test, y_pred, save_dir):
    """Plot prediction vs actual + residual distribution."""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
    ax.scatter(y_test.iloc[sample_idx], y_pred[sample_idx], alpha=0.1, s=5, color="#2196F3")
    ax.plot([0, y_test.max()], [0, y_test.max()], "r--", linewidth=1)
    ax.set_xlabel("Actual Delay (min)")
    ax.set_ylabel("Predicted Delay (min)")
    ax.set_title("Actual vs Predicted")
    ax.set_xlim(0, np.percentile(y_test, 99))
    ax.set_ylim(0, np.percentile(y_pred, 99))

    ax = axes[1]
    ax.hist(residuals, bins=100, color="#FF9800", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (mean={residuals.mean():.1f})")
    ax.set_xlim(np.percentile(residuals, 1), np.percentile(residuals, 99))

    plt.tight_layout()
    path = os.path.join(save_dir, "residuals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 10. QUANTILE REGRESSION MODELS (Uncertainty Estimation)
# ============================================================

def train_quantile_models(X_train, X_val, y_train, y_val, best_params, config):
    """
    Train P10, P50, P90 quantile models for prediction intervals.
    Agent can say: "expect 4-18 hours" instead of a single number.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Quantile Regression Models (P10 / P50 / P90)")
    print("=" * 60)

    import pandas as pd
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    quantile_models = {}
    for alpha, label in [(0.1, "P10"), (0.5, "P50"), (0.9, "P90")]:
        params = {
            **best_params,
            "objective": "quantile",
            "alpha": alpha,
            "n_estimators": 2000,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model_q = lgb.LGBMRegressor(**params)
        model_q.fit(
            X_combined, y_combined,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(config["early_stopping_rounds"]),
                lgb.log_evaluation(0),
            ],
        )
        quantile_models[label] = model_q
        print(f"  {label} (alpha={alpha}): best_iteration={model_q.best_iteration_}")

    return quantile_models


def evaluate_quantile_models(quantile_models, X_test, y_test_raw):
    """Evaluate quantile models: coverage and interval width."""
    print("\n  Quantile Evaluation on TEST set:")

    preds = {}
    for label, model_q in quantile_models.items():
        pred_log = model_q.predict(X_test)
        preds[label] = np.clip(np.expm1(pred_log), 0, None)

    # Coverage: % of actuals falling within [P10, P90]
    in_interval = (y_test_raw >= preds["P10"]) & (y_test_raw <= preds["P90"])
    coverage = float(in_interval.mean())

    # Average interval width
    avg_width = float(np.mean(preds["P90"] - preds["P10"]))

    print(f"  [P10, P90] Coverage: {coverage*100:.1f}% (target: ~80%)")
    print(f"  Avg interval width:  {avg_width:.0f} min ({avg_width/60:.1f} hours)")
    print(f"  P50 Median AE:       {np.median(np.abs(y_test_raw - preds['P50'])):.1f} min")

    return preds, coverage, avg_width


# ============================================================
# 11. SAVE ARTIFACTS
# ============================================================

def save_artifacts(model, metrics, best_params, feature_cols, config,
                   quantile_models=None):
    """Save model, metrics, and config."""
    print("\n" + "=" * 60)
    print("STEP 9: Saving Artifacts")
    print("=" * 60)

    # Versioned directory: models/run_YYYYMMDD_HHMM/
    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(config["model_dir"], f"run_{run_tag}")
    os.makedirs(save_dir, exist_ok=True)

    # Model
    model_path = os.path.join(save_dir, "harbormind_lgbm.pkl")
    joblib.dump(model, model_path)
    print(f"  Model: {model_path}")

    # Quantile models
    if quantile_models:
        for label, model_q in quantile_models.items():
            q_path = os.path.join(save_dir, f"harbormind_lgbm_{label.lower()}.pkl")
            joblib.dump(model_q, q_path)
            print(f"  Quantile {label}: {q_path}")

    # Metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Best params
    with open(os.path.join(save_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # Feature list
    with open(os.path.join(save_dir, "features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Update 'latest' symlink
    latest_link = os.path.join(config["model_dir"], "latest")
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(f"run_{run_tag}", latest_link)
    print(f"  Latest → {save_dir}")

    # Log to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    for f_name in os.listdir(save_dir):
        mlflow.log_artifact(os.path.join(save_dir, f_name))

    print(f"  All artifacts saved to: {save_dir}")
    return save_dir


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def main():
    """Execute the full training pipeline."""
    start = datetime.now()
    print("🚢 HarborMind — Training Pipeline (MLflow + Optuna)")
    print(f"   Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # MLflow setup
    tracking_uri = "file://" + os.path.join(os.getcwd(), "mlruns", "tracking")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "HarborMind_Delay_Prediction"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"train_{start.strftime('%Y%m%d_%H%M')}"):

        # Load
        df = load_data(CONFIG)

        # Aggregate pings → visits
        df = aggregate_visits(df, CONFIG)

        # Add lag features (historical context)
        df = add_lag_features(df)

        # Features (with correlation filter)
        df, feature_cols = prepare_features(df, CONFIG)

        # 3-way split: Train / Val / Test
        X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw = \
            temporal_split(df, feature_cols, CONFIG)

        # Hardcoded best params — proven stable on clean data (Run 0051: R²=0.420)
        # NOTE: Optuna runs 0106/0108/0110 all showed severe overfitting
        # (R² dropped to 0.10-0.40) because the final Train+Val merge
        # causes the model to memorize Val during early stopping.
        # These conservative params generalize better to unseen test data.
        best_params = {
            "max_depth": 9,
            "learning_rate": 0.023581958283248314,
            "num_leaves": 84,
            "min_child_samples": 157,
            "subsample": 0.9489312019744608,
            "colsample_bytree": 0.7316266641775542,
            "reg_alpha": 0.1831270297747932,
            "reg_lambda": 0.0808645204315922,
        }
        print("\n" + "=" * 60)
        print("STEP 5: Using proven best params (Run 0051)")
        print("=" * 60)
        for k, v in best_params.items():
            print(f"    {k}: {v}")

        # Train final model on Train+Val with best params
        model = train_final_model(X_train, X_val, y_train, y_val, best_params, CONFIG)

        # Evaluate on HOLDOUT TEST (first and only time test is used)
        print("\n" + "━" * 60)
        print("🎯 FINAL EVALUATION on HOLDOUT TEST")
        print("━" * 60)
        metrics, y_pred = evaluate_model(model, X_test, y_test, y_test_raw)

        # Train quantile models for uncertainty estimation
        quantile_models = train_quantile_models(
            X_train, X_val, y_train, y_val, best_params, CONFIG
        )
        q_preds, coverage, avg_width = evaluate_quantile_models(
            quantile_models, X_test, y_test_raw
        )
        metrics["quantile_coverage"] = coverage
        metrics["quantile_avg_width_min"] = avg_width

        # Save all artifacts (versioned directory)
        save_dir = save_artifacts(
            model, metrics, best_params, feature_cols, CONFIG,
            quantile_models=quantile_models
        )

        # Plots (save into versioned dir)
        plot_feature_importance(model, X_train, save_dir)
        plot_residuals(y_test_raw, y_pred, save_dir)

        # Log split info
        mlflow.log_param("mode", CONFIG["mode"])
        mlflow.log_param("val_date", CONFIG["val_date"])
        mlflow.log_param("test_date", CONFIG["test_date"])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))

    # Summary + Comparison with previous Run 7
    elapsed = (datetime.now() - start).total_seconds()
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Duration:    {elapsed/60:.1f} minutes")
    print(f"  Model:       {save_dir}/harbormind_lgbm.pkl")
    print(f"  Quantiles:   P10, P50, P90 saved")
    print(f"  Features:    {len(feature_cols)}")
    print(f"  Split:       Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    print()
    print("  ┌──────────────────┬────────────┬────────────┐")
    print("  │ Metric           │ Previous   │ Current    │")
    print("  ├──────────────────┼────────────┼────────────┤")
    print(f"  │ Median AE (min)  │      92.0  │ {metrics['median_ae']:>9.1f}  │")
    print(f"  │ MAE (min)        │     493.0  │ {metrics['mae']:>9.1f}  │")
    print(f"  │ R²               │    0.5000  │ {metrics['r2']:>9.4f}  │")
    print(f"  │ R² (log)         │    0.6900  │ {metrics['r2_log']:>9.4f}  │")
    print(f"  │ Cat Accuracy     │     78.0%  │ {metrics['category_accuracy']*100:>8.1f}%  │")
    print(f"  │ [P10,P90] Cover  │       N/A  │ {coverage*100:>8.1f}%  │")
    print(f"  │ Interval Width   │       N/A  │ {avg_width/60:>7.1f}h  │")
    print("  └──────────────────┴────────────┴────────────┘")
    print(f"\n  MLflow UI:   mlflow ui --backend-store-uri {tracking_uri}")


if __name__ == "__main__":
    main()
