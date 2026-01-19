# Network Data Exfiltration Detector (Clustering Approach)

## Project Overview

This project implements an unsupervised machine learning model to detect potential data exfiltration in network traffic. Using **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**, the system identifies anomalous flows that deviate from normal behavioral patterns.

The workflow follows a structured process, including automated preprocessing, hyperparameter tuning, and model interpretation through surrogate models and SHAP values.

---

## Project Structure

```text
├── data/               # Local storage for the 'merged.csv' dataset
├── notebooks/          # Jupyter notebooks (pobl_ml.ipynb)
├── models/             # Exported models, scalers, and metadata
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies

```

---

## Prerequisites

* **Python 3.13+**
* **AWS CLI** (configured to access the project S3 bucket)

---

## Setup and Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MACC-PBL1/IA
cd IA

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## Data Pipeline: S3 Integration

### 1. Download Data

The dataset used for training is a pre-processed, pre-merged file named `merged.csv` stored in S3. This file includes both flowmeter and connection logs.

Download it into the `data/` directory and rename it to `dataset.csv` for the notebook to recognize it:

```bash
aws s3 cp s3://your-bucket-name/merged.csv data/dataset.csv

```

### 2. Preprocessing & Training

Run the `pobl_ml.ipynb` notebook. The script performs:

 
**Basic Preprocessing:** Handles nulls, infinities, and removes non-predictive columns (IPs, IDs, Timestamps).


 
**Advanced Preprocessing:** Resolves multicollinearity by dropping highly correlated features (|r| > 0.8) and scales data using `RobustScaler` to preserve outliers.


 
**Model Selection:** Compares K-Means, Ward, and HDBSCAN, ultimately selecting HDBSCAN for its superior ability to identify noise as potential attacks.



---

## Model Persistence & Production

### 1. Local Export

Upon completion, the notebook automatically saves the following artifacts into the `models/` folder with a unique timestamp:

* `hdbscan_exfiltration_detector_TIMESTAMP.pkl`: The trained clustering model.
* `robust_scaler_TIMESTAMP.pkl`: The fitted scaler required for new data.
* `model_metadata_TIMESTAMP.json`: Performance metrics (DBCV score, noise percentage) and hyperparameters.



### 2. Upload to S3

Once you have identified a model suitable for production, upload the artifacts back to the S3 bucket to be consumed by the deployment pipeline:

```bash
aws s3 cp models/ s3://your-bucket-name/production_models/ --recursive

```

---

## Interpretability and Validation

For knowledge extraction and model validation:

* **Global Importance:** A Random Forest surrogate model identifies which network features (e.g., payload size, flow duration) drive the detection of anomalies.


* **Local Explanations:** SHAP analysis provides a "beeswarm" plot to explain why specific instances were flagged as exfiltration.


* **Statistical Validation:** The notebook uses the Hopkins Statistic to prove clustering tendency and the Elbow/Gap Statistic methods to justify the optimal cluster count.



---

