"""
HarborMind — Model Improvement Experiments
============================================
Tries 3 strategies to improve R² beyond 0.42:
  A) No-Merge: Train only, Val for early stopping (no data leakage)
  B) Enhanced Features: interaction features + tighter outlier cap
  C) Combined: A + B

Usage: python mlruns/experiment_compare.py
"""

import os, json, numpy as np, polars as pl, lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "Data-Engineering/data/processed/ais_2023_2025_clean.parquet"
VAL_DATE = "2024-07-01"
TEST_DATE = "2025-01-01"
TARGET = "delay_minutes"

BEST_PARAMS = {
    "max_depth": 9,
    "learning_rate": 0.023581958283248314,
    "num_leaves": 84,
    "min_child_samples": 157,
    "subsample": 0.9489312019744608,
    "colsample_bytree": 0.7316266641775542,
    "reg_alpha": 0.1831270297747932,
    "reg_lambda": 0.0808645204315922,
}

# ============================================================
# DATA LOADING & VISIT AGGREGATION (shared)
# ============================================================
def load_and_aggregate(enhanced=False):
    print("Loading data...")
    df = pl.read_parquet(DATA_PATH)
    df = df.filter(pl.col(TARGET).is_not_null())
    print(f"  {len(df):,} labeled pings")

    common_aggs = [
        pl.col("draft").first(),
        pl.col("length").first(),
        pl.col("width").first(),
        pl.col("vessel_type").first(),
        pl.col("vessel_area").first(),
        pl.col("dimension_ratio").first(),
        pl.col("draft_to_length_ratio").first(),
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
        pl.col(TARGET).first(),
    ]

    retro_aggs = [
        pl.col("sog").mean().alias("avg_sog"),
        pl.col("sog").max().alias("max_sog"),
        pl.col("sog").std().alias("std_sog"),
        pl.col("cog").std().alias("std_cog"),
        pl.col("acceleration").mean().alias("avg_accel"),
        pl.col("acceleration").std().alias("std_accel"),
        pl.col("heading_error").mean().alias("avg_heading_error"),
        pl.col("distance_to_port").min().alias("min_distance"),
        pl.col("distance_to_port").mean().alias("avg_distance"),
        pl.col("distance_to_port").first().alias("first_distance"),
        pl.col("ship_density").mean().alias("avg_density"),
        pl.col("ship_density").max().alias("max_density"),
        pl.col("avg_port_speed").mean().alias("avg_port_speed"),
        pl.col("port_throughput").mean().alias("avg_throughput"),
        pl.col("is_in_waiting_area").mean().alias("pct_waiting"),
        pl.col("wind_speed_10m").mean().alias("avg_wind"),
        pl.col("wind_speed_10m").max().alias("max_wind"),
        pl.col("wind_gusts_10m").max().alias("max_gusts"),
        pl.col("precipitation").sum().alias("total_precip"),
        pl.col("wave_height").mean().alias("avg_wave"),
        pl.col("swell_wave_height").mean().alias("avg_swell"),
        pl.len().alias("n_pings"),
    ]

    visits = df.group_by(["mmsi", "visit_id"]).agg(common_aggs + retro_aggs)

    # Lag features
    visits = visits.sort("arrival_time")
    for window in [5, 10, 20]:
        visits = visits.with_columns(
            pl.col(TARGET).shift(1)
            .rolling_mean(window_size=window, min_samples=1)
            .alias(f"rolling_delay_{window}")
        )

    arrival_ts = visits["arrival_time"]
    n = len(visits)
    arr_24, arr_48 = [], []
    for i in range(n):
        ct = arrival_ts[i]
        if ct is None:
            arr_24.append(None); arr_48.append(None); continue
        c24, c48 = 0, 0
        for j in range(i-1, max(i-200, -1), -1):
            if arrival_ts[j] is None: continue
            dh = (ct - arrival_ts[j]).total_seconds() / 3600
            if dh > 48: break
            c48 += 1
            if dh <= 24: c24 += 1
        arr_24.append(c24); arr_48.append(c48)

    visits = visits.with_columns([
        pl.Series("arrivals_24h", arr_24).cast(pl.Int32),
        pl.Series("arrivals_48h", arr_48).cast(pl.Int32),
    ])

    visits = visits.with_columns(
        pl.col("avg_density").shift(1)
        .rolling_mean(window_size=5, min_samples=1)
        .alias("recent_density_trend")
    )
    visits = visits.with_columns(
        pl.col(TARGET).shift(1)
        .rolling_mean(window_size=50, min_samples=1)
        .over("vessel_type")
        .alias("vessel_type_avg_delay")
    )
    visits = visits.with_columns(
        pl.col(TARGET).shift(1).over("mmsi").alias("prev_ship_delay")
    )

    # ---- ENHANCED FEATURES (Strategy B) ----
    if enhanced:
        visits = visits.with_columns([
            # Interaction: big ship in crowded port
            (pl.col("avg_density") * pl.col("vessel_area")).alias("density_x_area"),
            # Speed at distance (approaching pattern)
            (pl.col("avg_sog") * pl.col("first_distance")).alias("sog_x_distance"),
            # Draft loading ratio (how full)
            (pl.col("draft") / (pl.col("width") + 1e-6)).alias("draft_width_ratio"),
            # Congestion pressure (density / throughput)
            (pl.col("avg_density") / (pl.col("avg_throughput") + 1e-6)).alias("congestion_pressure"),
            # Weather severity composite
            (pl.col("avg_wind") * pl.col("avg_swell")).alias("weather_severity"),
            # Historical volatility (rolling std of delays)
            pl.col(TARGET).shift(1)
            .rolling_std(window_size=10, min_samples=3)
            .alias("rolling_delay_std_10"),
            # Distance range during visit
            (pl.col("first_distance") - pl.col("min_distance")).alias("distance_range"),
            # Vessel size category
            pl.when(pl.col("vessel_area") < 1000).then(pl.lit(0))
            .when(pl.col("vessel_area") < 5000).then(pl.lit(1))
            .when(pl.col("vessel_area") < 15000).then(pl.lit(2))
            .otherwise(pl.lit(3))
            .alias("size_category"),
        ])
        print(f"  Enhanced: +8 features added")

    print(f"  {len(visits):,} visits, {len(visits.columns)} columns")
    return visits


def prepare_and_split(visits, cap_hours=72):
    """Feature prep + temporal split."""
    drop_cols = ["mmsi", "visit_id", "arrival_time", TARGET]
    feature_cols = [c for c in visits.columns if c not in drop_cols]

    # Bool → int
    for col in feature_cols:
        if visits[col].dtype == pl.Boolean:
            visits = visits.with_columns(pl.col(col).cast(pl.Int8))

    # Corr filter
    numeric_df = visits.select(feature_cols).to_pandas().select_dtypes(include=["number"])
    corr = numeric_df.corr().abs()
    dropped = set()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i, j] > 0.95:
                ci, cj = corr.columns[i], corr.columns[j]
                drop_col = cj if corr[cj].mean() >= corr[ci].mean() else ci
                dropped.add(drop_col)
    feature_cols = [c for c in feature_cols if c not in dropped]

    # Split
    cap = cap_hours * 60
    visits = visits.with_columns(pl.col("arrival_time").dt.replace_time_zone(None))
    val_dt = datetime.strptime(VAL_DATE, "%Y-%m-%d")
    test_dt = datetime.strptime(TEST_DATE, "%Y-%m-%d")

    df_train = visits.filter(pl.col("arrival_time") < val_dt)
    df_val = visits.filter((pl.col("arrival_time") >= val_dt) & (pl.col("arrival_time") < test_dt))
    df_test = visits.filter(pl.col("arrival_time") >= test_dt)

    def _extract(part):
        X = part.select(feature_cols).to_pandas()
        for col in X.select_dtypes(include=['object', 'string']).columns:
            X[col] = X[col].astype('category')
        y_raw = part[TARGET].to_pandas().clip(upper=cap)
        y_log = np.log1p(y_raw)
        return X, y_log, y_raw

    X_tr, y_tr, y_tr_raw = _extract(df_train)
    X_val, y_val, y_val_raw = _extract(df_val)
    X_te, y_te, y_te_raw = _extract(df_test)

    return X_tr, X_val, X_te, y_tr, y_val, y_te, y_te_raw, feature_cols


def evaluate(model, X_test, y_test_log, y_test_raw):
    """Evaluate and return metrics dict."""
    y_pred_log = model.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    y_test = y_test_raw

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "median_ae": float(np.median(np.abs(y_test - y_pred))),
        "r2_log": float(r2_score(y_test_log, y_pred_log)),
    }

    bins = [0, 60, 360, 1440, float("inf")]
    actual_cat = np.digitize(y_test, bins) - 1
    pred_cat = np.digitize(y_pred, bins) - 1
    metrics["cat_accuracy"] = float(np.mean(actual_cat == pred_cat))

    return metrics


# ============================================================
# STRATEGY A: No-Merge (Train only, Val for early stopping)
# ============================================================
def strategy_a(X_tr, X_val, X_te, y_tr, y_val, y_te, y_te_raw):
    print("\n" + "=" * 60)
    print("STRATEGY A: No-Merge (Train only, Val for early stopping)")
    print("=" * 60)

    params = {**BEST_PARAMS, "n_estimators": 3000, "random_state": 42, "n_jobs": -1, "verbose": -1}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return evaluate(model, X_te, y_te, y_te_raw)


# ============================================================
# STRATEGY B: Enhanced Features + Tighter Cap
# ============================================================
def strategy_b(X_tr, X_val, X_te, y_tr, y_val, y_te, y_te_raw):
    print("\n" + "=" * 60)
    print("STRATEGY B: Enhanced Features (interaction + size bins)")
    print("=" * 60)

    import pandas as pd
    X_comb = pd.concat([X_tr, X_val], ignore_index=True)
    y_comb = pd.concat([y_tr, y_val], ignore_index=True)

    params = {**BEST_PARAMS, "n_estimators": 2000, "random_state": 42, "n_jobs": -1, "verbose": -1}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_comb, y_comb,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return evaluate(model, X_te, y_te, y_te_raw)


# ============================================================
# STRATEGY C: No-Merge + Enhanced Features + Cap 48h
# ============================================================
def strategy_c(X_tr, X_val, X_te, y_tr, y_val, y_te, y_te_raw):
    print("\n" + "=" * 60)
    print("STRATEGY C: No-Merge + Enhanced + Cap 48h")
    print("=" * 60)

    params = {**BEST_PARAMS, "n_estimators": 3000, "random_state": 42, "n_jobs": -1, "verbose": -1}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return evaluate(model, X_te, y_te, y_te_raw)


# ============================================================
# MAIN
# ============================================================
def main():
    start = datetime.now()
    print("🧪 HarborMind — Model Improvement Experiments")
    print(f"   {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}

    # --- Baseline (current: merged, 72h cap, standard features) ---
    print("━" * 60)
    print("BASELINE: Current model (merged Train+Val, cap=72h)")
    print("━" * 60)
    visits_std = load_and_aggregate(enhanced=False)
    X_tr, X_val, X_te, y_tr, y_val, y_te, y_te_raw, feat = prepare_and_split(visits_std, cap_hours=72)
    print(f"  Train={len(X_tr)} Val={len(X_val)} Test={len(X_te)} Features={len(feat)}")

    import pandas as pd
    X_comb = pd.concat([X_tr, X_val], ignore_index=True)
    y_comb = pd.concat([y_tr, y_val], ignore_index=True)
    params = {**BEST_PARAMS, "n_estimators": 2000, "random_state": 42, "n_jobs": -1, "verbose": -1}
    model_base = lgb.LGBMRegressor(**params)
    model_base.fit(X_comb, y_comb, eval_set=[(X_val, y_val)], eval_metric="mae",
                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    results["Baseline"] = evaluate(model_base, X_te, y_te, y_te_raw)

    # --- Strategy A: No-Merge ---
    results["A: No-Merge"] = strategy_a(X_tr, X_val, X_te, y_tr, y_val, y_te, y_te_raw)

    # --- Strategy B: Enhanced features (need to reload with enhanced=True) ---
    print("\n━" * 60)
    print("Loading enhanced features...")
    visits_enh = load_and_aggregate(enhanced=True)
    X_tr_e, X_val_e, X_te_e, y_tr_e, y_val_e, y_te_e, y_te_raw_e, feat_e = prepare_and_split(visits_enh, cap_hours=72)
    print(f"  Train={len(X_tr_e)} Val={len(X_val_e)} Test={len(X_te_e)} Features={len(feat_e)}")
    results["B: Enhanced"] = strategy_b(X_tr_e, X_val_e, X_te_e, y_tr_e, y_val_e, y_te_e, y_te_raw_e)

    # --- Strategy C: No-Merge + Enhanced + Cap 48h ---
    print("\n━" * 60)
    print("Loading enhanced features with 48h cap...")
    X_tr_c, X_val_c, X_te_c, y_tr_c, y_val_c, y_te_c, y_te_raw_c, feat_c = prepare_and_split(visits_enh, cap_hours=48)
    results["C: NoMerge+Enh+48h"] = strategy_c(X_tr_c, X_val_c, X_te_c, y_tr_c, y_val_c, y_te_c, y_te_raw_c)

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    elapsed = (datetime.now() - start).total_seconds()
    print("\n\n" + "=" * 80)
    print("📊 EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<25} {'R²':>8} {'R²(log)':>8} {'MedAE':>8} {'MAE':>8} {'CatAcc':>8}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<25} {m['r2']:>8.4f} {m['r2_log']:>8.4f} {m['median_ae']:>7.1f}m {m['mae']:>7.1f}m {m['cat_accuracy']*100:>7.1f}%")

    # Find best
    best_name = max(results, key=lambda k: results[k]["r2"])
    best = results[best_name]
    baseline = results["Baseline"]
    delta = best["r2"] - baseline["r2"]

    print(f"\n🏆 Best: {best_name} (R²={best['r2']:.4f}, Δ={delta:+.4f} vs Baseline)")
    print(f"   Duration: {elapsed/60:.1f} minutes")

    # Save results
    save_path = "mlruns/models/experiment_results.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved: {save_path}")


if __name__ == "__main__":
    main()
