import polars as pl
import os

data_path = "Data-Engineering/data/processed/ais_2023_2025.parquet"
if not os.path.exists(data_path):
    print(f"File not found: {data_path}")
    exit(1)

df = pl.read_parquet(data_path)

print("="*50)
print("DATASET QUALITY CHECK")
print("="*50)

print(f"1. Total Rows: {df.shape[0]:,}")
print(f"2. Total Columns: {df.shape[1]}")

print("\n3. TIME RANGE")
print(f"   Start: {df['base_date_time'].min()}")
print(f"   End:   {df['base_date_time'].max()}")

print("\n4. IDENTIFIERS")
print(f"   Unique Ships (MMSI): {df['mmsi'].n_unique():,}")
unique_visits = df.select(['mmsi', 'visit_id']).unique().shape[0]
print(f"   Unique Visits:       {unique_visits:,}")

print("\n5. KEY FEATURES STATISTICS")
desc = df.select(['delay_minutes', 'ship_density', 'port_throughput', 'sog', 'distance_to_port', 'wind_speed_10m']).describe()
print(desc)

print("\n6. MISSING VALUES (>0%)")
nulls = df.null_count()
for col in df.columns:
    count = nulls[col][0]
    if count > 0:
        pct = (count / df.shape[0]) * 100
        print(f"   - {col}: {count:,} ({pct:.2f}%)")

print("\n7. LABEL DISTRIBUTION (delay_minutes)")
labeled = df.filter(pl.col("delay_minutes").is_not_null())
print(f"   Labeled Rows: {len(labeled):,} ({len(labeled)/df.shape[0]*100:.2f}%)")
print(f"   Unique visits with labels: {labeled['visit_id'].n_unique():,}")
if len(labeled) > 0:
    print("   Percentiles (min):")
    for p in [0.25, 0.5, 0.75, 0.9, 0.99]:
        val = labeled["delay_minutes"].quantile(p)
        print(f"      {int(p*100)}th: {val:.1f} min ({val/60:.1f} h)")

print("\n8. SANITY CHECKS")
moored = df.filter(pl.col("sog") < 0.5)
print(f"   Rows with SOG < 0.5 (moored/waiting): {len(moored):,} ({len(moored)/df.shape[0]*100:.2f}%)")
