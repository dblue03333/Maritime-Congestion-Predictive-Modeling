import pandas as pd
import numpy as np
import os

def generate_synthetic_data_1st_version(num_records=500):
    """
    Generate synthetic AIS data for port congestion prediction.

    This function simulates realistic port operations and environmental factors
    to create a dataset for training and evaluating congestion prediction models.
    The 'delay_minutes' and 'congestion_label' are simulated as ground truth
    based on a set of predefined rules and random noise, reflecting how
    congestion might manifest in a real-world scenario.

    Features:
    - arrival_vessels_24h: Number of vessels arriving in the last 24h.
    - port_capacity_utilization: Current occupancy percentage (0-1).
    - visibility_km: Weather visibility (lower means higher delay).
    - wind_speed_knots: High winds can slow berthing.
    - queue_length: Number of vessels waiting for a berth.
    - vessel_type: (0: Cargo, 1: Tanker, 2: Container).
    - arrival_hour: (0-23) for temporal patterns.
    - day_of_week: (0-6, Monday-Sunday) for weekly patterns.
    - is_monsoon_season: Binary (0: No, 1: Yes) indicating seasonal impact.

    Ground Truth Targets:
    - delay_minutes: Numerical target for prediction, representing total delay.
    - congestion_label: Categorical label (0: Low, 1: Moderate, 2: High Congestion/Delay).
    - confidence_score: Simulated ground truth confidence (for training/eval).
    """

    np.random.seed(42)

    # Features
    arrival_vessels_24h = np.random.randint(5, 50, size=num_records)
    port_capacity_utilization = np.random.uniform(0.1, 0.95, size=num_records)
    visibility_km = np.random.uniform(1.0, 20.0, size=num_records)
    wind_speed_knots = np.random.uniform(5.0, 45.0, size=num_records)
    queue_length = np.random.randint(0, 15, size=num_records)
    arrival_hour = np.random.randint(0, 24, size=num_records)
    vessel_type = np.random.choice([0, 1, 2], size=num_records, p=[0.4, 0.4, 0.2])
    day_of_week = np.random.randint(0, 7, size=num_records) # 0=Monday, 6=Sunday
    is_monsoon_season = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2]) # 20% chance of monsoon

    # calculate delay with stronger explicit rules to make correlation heatmap pop
    # vessel type: Container (2) cause highest base delays, Cargo (0) medium, Tanker (1) low
    base_delay = np.where(vessel_type == 2, 60, np.where(vessel_type == 1, 15, 30))
    
    delay = base_delay + \
            (arrival_vessels_24h * 2.0) + \
            (port_capacity_utilization * 100) + \
            (queue_length * 6)
    
    # environmental factors (Linear instead of thresholds to improve Pearson correlation)
    # wind speed adds linear delay directly
    delay += (wind_speed_knots * 1.5) 
    # visibility has an inverse relationship (max visibility is 20, so 20 - visibility means lower visibility = higher penalty)
    delay += ((20.0 - visibility_km) * 3.0) 
    
    # temporal patterns
    # keep some threshold correlations but make them slightly larger
    delay += np.where((arrival_hour >= 1) & (arrival_hour <= 5), 30, 0)
    delay += np.where((day_of_week >= 5), 25, 0) 
    
    # seasonal impact
    delay += is_monsoon_season * 40

    # add smaller random noise so the signal-to-noise ratio is really strong
    delay += np.random.normal(0, 5, size=num_records)

    # ensure delay is non-negative
    delay = np.maximum(0, delay)

    # binary label: >240 min = financially impactful congestion (demurrage threshold)
    congestion_label = (delay > 240).astype(int)

    # 3-class label for richer analysis / reporting
    #   0 = No Congestion  (≤ 150 min)
    #   1 = Moderate       (150–300 min)
    #   2 = High           (> 300 min)
    congestion_level = np.where(delay <= 150, 0, np.where(delay <= 300, 1, 2))

    # confidence score: higher = prediction is far from the 240-min decision boundary
    confidence_score = 0.6 + (np.abs(delay - 240) / 400) * 0.39
    confidence_score = np.clip(confidence_score, 0.6, 0.99)


    df = pd.DataFrame({
        # ── features (model inputs) ──
        'arrival_vessels_24h':       arrival_vessels_24h,
        'port_capacity_utilization': port_capacity_utilization.round(4),
        'visibility_km':             visibility_km.round(2),
        'wind_speed_knots':          wind_speed_knots.round(2),
        'queue_length':              queue_length,
        'arrival_hour':              arrival_hour,
        'vessel_type':               vessel_type,          # 0=Container, 1=Tanker, 2=Bulk
        'is_monsoon_season':         is_monsoon_season,    # 1 if May–Oct
        'day_of_week':               day_of_week,          # 0=Mon … 6=Sun
        # ── Targets / Ground Truth ──
        'delay_minutes':             delay.round(2),
        'congestion_label':          congestion_label,     # binary  (primary XGBoost target)
        'congestion_level':          congestion_level,     # 3-class (for reporting)
        'confidence_score':          confidence_score.round(4),
    })

    # Save
    output_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f'synthetic_port_data_v1_{num_records}.csv')
    df.to_csv(output_path, index=False)

    print(f"Generated {num_records} records → {output_path}")
    print(f"\nBinary Label Distribution (congestion_label):")
    print(f"   0 — No/Low congestion (≤240 min) : {(df['congestion_label']==0).sum()} ({(df['congestion_label']==0).mean()*100:.1f}%)")
    print(f"   1 — Congestion        (>240 min) : {(df['congestion_label']==1).sum()} ({(df['congestion_label']==1).mean()*100:.1f}%)")
    print(f"\n3-Class Distribution (congestion_level):")
    print(f"   0 — No Congestion   (≤120 min) : {(df['congestion_level']==0).sum()}")
    print(f"   1 — Moderate        (≤360 min) : {(df['congestion_level']==1).sum()}")
    print(f"   2 — High Congestion (>360 min) : {(df['congestion_level']==2).sum()}")
    print(f"\nDelay Statistics (delay_minutes):")
    print(df['delay_minutes'].describe().round(1).to_string())

    return df


def generate_synthetic_data_2nd_version(num_records=500):
    """
    Generate synthetic AIS data for port congestion prediction.

    This function simulates realistic port operations and environmental factors
    to create a dataset for training and evaluating congestion prediction models.
    The 'delay_minutes' and 'congestion_label' are simulated as ground truth
    based on a set of predefined rules and random noise, reflecting how
    congestion might manifest in a real-world scenario.

    Features:
    - arrival_vessels_24h: Number of vessels arriving in the last 24h.
    - port_capacity_utilization: Current occupancy percentage (0-1).
    - visibility_km: Weather visibility (lower means higher delay).
    - wind_speed_knots: High winds can slow berthing.
    - queue_length: Number of vessels waiting for a berth.
    - vessel_type: (0: Cargo, 1: Tanker, 2: Container).
    - arrival_hour: (0-23) for temporal patterns.
    - day_of_week: (0-6, Monday-Sunday) for weekly patterns.
    - is_monsoon_season: Binary (0: No, 1: Yes) indicating seasonal impact.

    Ground Truth Targets:
    - delay_minutes: Numerical target for prediction, representing total delay.
    - congestion_label: Categorical label (0: Low, 1: Moderate, 2: High Congestion/Delay).
    - confidence_score: Simulated ground truth confidence (for training/eval).
    """

    np.random.seed(42)

    # Features
    arrival_vessels_24h = np.random.randint(5, 50, size=num_records)
    port_capacity_utilization = np.random.uniform(0.1, 0.95, size=num_records)
    visibility_km = np.random.uniform(1.0, 20.0, size=num_records)
    wind_speed_knots = np.random.uniform(5.0, 45.0, size=num_records)
    queue_length = np.random.randint(0, 15, size=num_records)
    arrival_hour = np.random.randint(0, 24, size=num_records)
    vessel_type = np.random.choice([0, 1, 2], size=num_records, p=[0.4, 0.4, 0.2])
    day_of_week = np.random.randint(0, 7, size=num_records) # 0=Monday, 6=Sunday
    is_monsoon_season = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2]) # 20% chance of monsoon

    # 1. The Base Processing Time (The Foundation)
    # Container (2) causes highest base delays, Cargo (0) medium, Tanker (1) low
    base_processing = np.where(vessel_type == 2, 60, np.where(vessel_type == 1, 15, 30)).astype(float)
    base_processing += (queue_length * 15.0)

    # 2. The Exponential Capacity Multiplier (Queueing Theory)
    exponential_multiplier = 1.0 + (port_capacity_utilization ** 3)
    delay = base_processing * exponential_multiplier
    
    # 3. The "Hard Stop" Environmental Triggers
    # If wind > 35 OR visibility < 1.5, port halts (+240 mins)
    # Otherwise, apply small linear penalty
    hard_stop_condition = (wind_speed_knots > 35.0) | (visibility_km < 1.5)
    environmental_penalty = np.where(
        hard_stop_condition, 
        240.0, 
        (wind_speed_knots * 1.5) + np.maximum(0, (20.0 - visibility_km) * 2.0)
    )
    delay += environmental_penalty
    
    # 4. The "Night Gridlock" Interaction (Cát Lái Truck Traffic simulation)
    # Between 22:00 and 04:00
    is_night = (arrival_hour >= 22) | (arrival_hour <= 4)
    night_gridlock_penalty = np.where(
        is_night & (port_capacity_utilization > 0.80),
        120.0,  # Severe gridlock
        np.where(is_night, 30.0, 0.0) # Standard night penalty
    )
    delay += night_gridlock_penalty
    
    # Temporal patterns and seasonal impact
    delay += np.where((day_of_week >= 5), 25, 0) # Weekend
    delay += is_monsoon_season * 40

    # Add random noise
    delay += np.random.normal(0, 10, size=num_records)

    # Ensure delay is non-negative
    delay = np.maximum(0, delay)

    # Binary label: >240 min = financially impactful congestion (demurrage threshold)
    congestion_label = (delay > 240).astype(int)

    # 3-class label for richer analysis / reporting
    #   0 = No Congestion  (≤ 150 min)
    #   1 = Moderate       (150–300 min)
    #   2 = High           (> 300 min)
    congestion_level = np.where(delay <= 150, 0, np.where(delay <= 300, 1, 2))

    # Confidence score: higher = prediction is far from the 240-min decision boundary
    confidence_score = 0.6 + (np.abs(delay - 240) / 400) * 0.39
    confidence_score = np.clip(confidence_score, 0.6, 0.99)


    df = pd.DataFrame({
        # ── Features (model inputs) ──
        'arrival_vessels_24h':       arrival_vessels_24h,
        'port_capacity_utilization': port_capacity_utilization.round(4),
        'visibility_km':             visibility_km.round(2),
        'wind_speed_knots':          wind_speed_knots.round(2),
        'queue_length':              queue_length,
        'arrival_hour':              arrival_hour,
        'vessel_type':               vessel_type,          # 0=Container, 1=Tanker, 2=Bulk
        'is_monsoon_season':         is_monsoon_season,    # 1 if May–Oct
        'day_of_week':               day_of_week,          # 0=Mon … 6=Sun
        # ── Targets / Ground Truth ──
        'delay_minutes':             delay.round(2),
        'congestion_label':          congestion_label,     # binary  (primary XGBoost target)
        'congestion_level':          congestion_level,     # 3-class (for reporting)
        'confidence_score':          confidence_score.round(4),
    })

    # Save
    output_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f'synthetic_port_data_v2_{num_records}.csv')
    df.to_csv(output_path, index=False)

    print(f"Generated {num_records} records → {output_path}")
    print(f"\nBinary Label Distribution (congestion_label):")
    print(f"   0 — No/Low congestion (≤240 min) : {(df['congestion_label']==0).sum()} ({(df['congestion_label']==0).mean()*100:.1f}%)")
    print(f"   1 — Congestion        (>240 min) : {(df['congestion_label']==1).sum()} ({(df['congestion_label']==1).mean()*100:.1f}%)")
    print(f"\n3-Class Distribution (congestion_level):")
    print(f"   0 — No Congestion   (≤120 min) : {(df['congestion_level']==0).sum()}")
    print(f"   1 — Moderate        (≤360 min) : {(df['congestion_level']==1).sum()}")
    print(f"   2 — High Congestion (>360 min) : {(df['congestion_level']==2).sum()}")
    print(f"\nDelay Statistics (delay_minutes):")
    print(df['delay_minutes'].describe().round(1).to_string())

    return df


if __name__ == "__main__":
    for n in [500, 5000, 50000, 500000]:
        print(f"\n==========================================")
        print(f"Generating v1 data for {n} records")
        print(f"==========================================")
        generate_synthetic_data_1st_version(num_records=n)
        
        print(f"\n==========================================")
        print(f"Generating v2 data for {n} records")
        print(f"==========================================")
        generate_synthetic_data_2nd_version(num_records=n)