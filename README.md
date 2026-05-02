# ⚓ HarborMind: Maritime Congestion Predictive Pipeline

![Python](https://img.shields.io/badge/python-3.10-blue) ![PySpark](https://img.shields.io/badge/PySpark-Data%20Engineering-E25A1C) ![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2) ![LightGBM](https://img.shields.io/badge/LightGBM-Predictive%20Model-yellow)

> **Note:** This repository contains the **Data Engineering** and **MLOps pipeline** that powers the backend of HarborMind. The Agentic UI (Gemma 4 integration) is hosted in a separate module.

## 📖 Overview
Global trade relies heavily on the efficiency of maritime ports. Congestion at major hubs like the Port of Long Beach leads to massive CO2 emissions from idling ships and disrupts global supply chains.

**HarborMind** is an end-to-end Machine Learning pipeline designed to forecast vessel delays using raw satellite tracking data. By calculating precise delay predictions with statistical confidence bounds, this pipeline serves as the "brain" for a downstream Multi-Agent decision-support system.

### 💡 Inspiration & Motivation
The inspiration for this project originated from an end-to-end technical assessment at Safiri AI ([safiri-port-congestion-ai](https://github.com/dblue03333/safiri-port-congestion-ai)). That initial test contained numerous flaws due to a lack of practical experience and real-world friction. Recognizing those gaps, I decided to build *this* project entirely from scratch. By selecting a highly reliable, massive raw dataset (NOAA MarineCadastre), I aimed to dive deep into data processing, feature engineering, and robust model training to maximize my learning and create a truly production-ready pipeline.

---

## 🏗️ 1. [Data Engineering (Lakehouse Architecture)](./Data-Engineering/)

Processing geospatial time-series data requires massive scale. The ETL pipeline was built using a **Medallion Architecture (Bronze/Silver/Gold)** via **PySpark on Microsoft Fabric (Data Lake)**.

**Highlights:**
- **Raw Data:** Processed **21.7M+ AIS terrestrial pings** spanning 30 months (June 2023 - Dec 2025).
- **Output:** 6,000+ structured vessel visits.
- **Features Engineered:** 54 complex kinetic, temporal, and spatial features.
- **Key Techniques:** Kinematic state derivation (acceleration, distance range), Cyclical temporal encoding, Open-Meteo weather integration.

👉 **[Read the full Data Engineering documentation →](./Data-Engineering/README.md)**

---

## 🧠 2. [MLOps & Predictive Modeling](./ML-Experiment/)

The predictive core utilizes **LightGBM**, optimized through a rigorous MLOps lifecycle tracked by **MLflow**. We focused heavily on **"Honest AI"** — ensuring predictions are robust against data leakage and accompanied by statistically sound uncertainty bounds.

**Highlights:**
- **Robust Validation:** Enforced an **Out-of-Time (OOT) strategy** (2023-2024 Train / 2025 Test) to completely eliminate time-series data leakage.
- **Quantile Regression:** Integrated P10/P50/P90 models to generate uncertainty bounds, achieving a **65.7% quantile coverage**.
- **Optimization:** Tracked 50+ Bayesian optimization trials via **Optuna**.
- **Final Metrics:** Achieved an **R² = 0.505** and a **Median Absolute Error (MedAE) of 95 minutes** on the 2025 holdout set.

👉 **[Read the full ML Experiment logs, failures, and evolution →](./ML-Experiment/README.md)**

---

## 🚀 How to Run (Step-by-Step)

### Option A: Use Pre-processed Data (Recommended for ML)

If you only want to train the model without re-running the entire Data Engineering pipeline:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DBlue03333/Maritime-Congestion-Predictive-Modeling.git
   cd Maritime-Congestion-Predictive-Modeling
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Mac/Linux:
   python3 -m venv venv
   source venv/bin/activate
   
   # Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pre-processed dataset from Kaggle:**
   ```bash
   cd Data-Engineering
   jupyter notebook download_kaggle_data.ipynb
   ```
   This notebook uses the Kaggle API to download `ais_2023_2025_clean.parquet` (~580MB) directly into `Data-Engineering/data/processed/`.

5. **Run the training pipeline:**
   ```bash
   cd ../ML-Experiment
   python train.py
   ```

### Option B: Full Pipeline Reproduction (From Raw NOAA Data)

If you want to reproduce the entire ETL pipeline from scratch, see the detailed instructions inside the [Data-Engineering/README.md](./Data-Engineering/README.md). This requires **Microsoft Fabric** (or Azure Synapse with PySpark + Delta Lake).

---
*Built by [Kelvin Nguyen](https://github.com/DBlue03333)*
