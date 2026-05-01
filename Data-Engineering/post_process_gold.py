"""
HarborMind - Gold Layer Post-Processing (Data Quality Fix)
===========================================================
Fixes identified during comprehensive DE review & verification:

1. Clip extreme acceleration outliers (GPS jitter artifacts)
2. Cap delay outliers at a physically reasonable maximum
3. Recalculate time_in_zone_hours PER VISIT (fix cross-visit accumulation bug)
4. Drop redundant columns
5. Verify primary key uniqueness
6. Export clean version

Run: /Users/kelvinnguyen/Projects/.venv/bin/python post_process_gold.py
"""

import polars as pl
import os
import time

INPUT  = "data/processed/ais_2023_2025.parquet"
OUTPUT = os.path.expanduser("~/Desktop/ais_2023_2025_clean.parquet")

# ── Thresholds (based on maritime physics) ──────────────────────────
ACCEL_MIN = -10.0   # knots/hour — max deceleration for cargo/tanker
ACCEL_MAX =  10.0   # knots/hour — max acceleration for cargo/tanker  
DELAY_MAX_HOURS = 24 * 30  # 30 days — beyond this = layup/maintenance, not congestion
COLS_TO_DROP = ["cargo"]   # Documented as redundant in AIS_Preprocessing.ipynb

def main():
    t0 = time.time()
    
    print("=" * 60)
    print("🔧 HARBORMIND GOLD LAYER POST-PROCESSING v2")
    print("=" * 60)
    
    # ── Load ────────────────────────────────────────────────────────
    print(f"\n📂 Loading {INPUT}...")
    df = pl.read_parquet(INPUT)
    original_shape = df.shape
    print(f"   Original shape: {df.shape}")
    
    # ── 1. Clip Acceleration Outliers ───────────────────────────────
    print("\n[1/6] Clipping acceleration outliers...")
    before = df.filter(
        pl.col("acceleration").is_not_null() &
        ((pl.col("acceleration") < ACCEL_MIN) | (pl.col("acceleration") > ACCEL_MAX))
    ).shape[0]
    
    df = df.with_columns(
        pl.col("acceleration").clip(ACCEL_MIN, ACCEL_MAX).alias("acceleration")
    )
    print(f"   ✅ Clipped {before:,} rows to [{ACCEL_MIN}, {ACCEL_MAX}] knots/hr")
    print(f"   New range: [{df['acceleration'].min()}, {df['acceleration'].max()}]")
    
    # ── 2. Cap Delay Outliers ──────────────────────────────────────
    print("\n[2/6] Capping delay outliers...")
    delay_max_min = DELAY_MAX_HOURS * 60
    
    before_extreme = df.filter(
        pl.col("delay_minutes").is_not_null() & (pl.col("delay_minutes") > delay_max_min)
    ).shape[0]
    
    df = df.with_columns(
        pl.when(pl.col("delay_minutes") > delay_max_min)
          .then(None)
          .otherwise(pl.col("delay_minutes"))
          .alias("delay_minutes")
    )
    print(f"   ✅ Nullified {before_extreme:,} rows with delay > {DELAY_MAX_HOURS}h ({DELAY_MAX_HOURS//24}d)")
    
    labeled = df.filter(pl.col("delay_minutes").is_not_null())
    print(f"   New P50/P75/P99: {labeled['delay_minutes'].quantile(0.5):.0f} / {labeled['delay_minutes'].quantile(0.75):.0f} / {labeled['delay_minutes'].quantile(0.99):.0f} min")
    
    # ── 3. FIX: Recalculate time_in_zone_hours PER VISIT ──────────
    # BUG: Original pipeline used Window.partitionBy("mmsi") which
    #       accumulates across visits. Visit 2+ starts with inflated
    #       values carried over from Visit 1. Must partition by
    #       (mmsi, visit_id) to reset per visit.
    print("\n[3/6] Recalculating time_in_zone_hours (per-visit fix)...")
    
    # Show the bug first
    sample_ship = (df.group_by("mmsi")
        .agg(pl.col("visit_id").n_unique().alias("n"))
        .filter(pl.col("n") > 2)
        .head(1)["mmsi"][0])
    
    bug_demo = df.filter(pl.col("mmsi") == sample_ship).sort("base_date_time")
    visits_before = []
    for vid in bug_demo["visit_id"].unique().sort().to_list()[:3]:
        v = bug_demo.filter(pl.col("visit_id") == vid)
        visits_before.append(f"Visit {vid}: first={v['time_in_zone_hours'][0]:.1f}h")
    print(f"   BEFORE (MMSI={sample_ship}): {' | '.join(visits_before)}")
    
    # Recalculate: cumulative stationary hours PER (mmsi, visit_id)
    df = df.sort(["mmsi", "visit_id", "base_date_time"])
    
    df = df.with_columns(
        # Time diff in hours between consecutive pings (within same mmsi+visit)
        # NOTE: Polars casts datetime to MICROSECONDS (not seconds), so divide by 1e6 first
        (
            pl.col("base_date_time").cast(pl.Int64) 
            - pl.col("base_date_time").shift(1).over("mmsi", "visit_id").cast(pl.Int64)
        ).truediv(1_000_000 * 3600).alias("_td_hours")
    )
    
    df = df.with_columns(
        # Only count hours when stationary (SOG < 0.5)
        pl.when((pl.col("sog") < 0.5) & pl.col("_td_hours").is_not_null() & (pl.col("_td_hours") > 0) & (pl.col("_td_hours") < 24))
          .then(pl.col("_td_hours"))
          .otherwise(0.0)
          .alias("_stat_hrs")
    )
    
    df = df.with_columns(
        # Cumulative sum PER (mmsi, visit_id) — THIS IS THE FIX
        pl.col("_stat_hrs").cum_sum().over("mmsi", "visit_id").alias("time_in_zone_hours")
    )
    
    df = df.drop(["_td_hours", "_stat_hrs"])
    
    # Show the fix
    fix_demo = df.filter(pl.col("mmsi") == sample_ship).sort("base_date_time")
    visits_after = []
    for vid in fix_demo["visit_id"].unique().sort().to_list()[:3]:
        v = fix_demo.filter(pl.col("visit_id") == vid)
        visits_after.append(f"Visit {vid}: first={v['time_in_zone_hours'][0]:.1f}h")
    print(f"   AFTER  (MMSI={sample_ship}): {' | '.join(visits_after)}")
    print(f"   ✅ time_in_zone_hours now resets to 0.0 at start of each visit")
    
    # ── 4. Drop Redundant Columns ──────────────────────────────────
    print("\n[4/6] Dropping redundant columns...")
    existing_drops = [c for c in COLS_TO_DROP if c in df.columns]
    if existing_drops:
        df = df.drop(existing_drops)
        print(f"   ✅ Dropped: {existing_drops}")
    else:
        print(f"   ℹ️  No columns to drop")
    
    # ── 5. Verify Primary Key Uniqueness ───────────────────────────
    print("\n[5/6] Verifying primary key uniqueness (mmsi + base_date_time)...")
    total = df.shape[0]
    unique = df.select(["mmsi", "base_date_time"]).unique().shape[0]
    dupes = total - unique
    
    if dupes > 0:
        print(f"   ⚠️  Found {dupes:,} duplicate rows! Removing...")
        df = df.unique(subset=["mmsi", "base_date_time"], keep="first")
        print(f"   ✅ After dedup: {df.shape[0]:,} rows")
    else:
        print(f"   ✅ No duplicates found. Primary key is clean.")
    
    # ── 6. Final Report & Save ─────────────────────────────────────
    print("\n[6/6] Final quality report...")
    print(f"   Shape: {original_shape} → {df.shape}")
    print(f"   Columns: {df.shape[1]}")
    
    labeled_final = df.filter(pl.col("delay_minutes").is_not_null())
    print(f"   Labeled rows: {labeled_final.shape[0]:,} ({labeled_final.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"   Labeled visits: {labeled_final.select(['mmsi','visit_id']).unique().shape[0]:,}")
    print(f"   SOG range: [{df['sog'].min()}, {df['sog'].max()}]")
    print(f"   Accel range: [{df['acceleration'].min()}, {df['acceleration'].max()}]")
    print(f"   time_in_zone min/max: {df['time_in_zone_hours'].min():.2f} / {df['time_in_zone_hours'].max():.2f}")
    
    # Save
    print(f"\n💾 Saving to {OUTPUT}...")
    df.write_parquet(OUTPUT)
    
    new_size = os.path.getsize(OUTPUT) / 1e6
    print(f"   ✅ Saved! Size: {new_size:.1f} MB")
    
    elapsed = time.time() - t0
    print(f"\n🏁 Done in {elapsed:.1f} seconds.")
    print("=" * 60)

if __name__ == "__main__":
    main()
