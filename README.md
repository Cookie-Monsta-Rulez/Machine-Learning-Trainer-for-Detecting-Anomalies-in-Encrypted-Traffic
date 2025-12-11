# Encrypted Traffic Anomaly Detection Using Flow-Based Machine Learning  
Random Forest Classifier on CIC-IDS2017/2018

## Overview
This project evaluates whether flow-based machine learning can detect anomalous activity in encrypted network traffic (TLS 1.2, TLS 1.3, HTTPS). Since encrypted traffic prevents payload inspection, the model relies entirely on metadata features such as flow duration, packet counts, byte statistics, and timing intervals.

A Random Forest classifier is trained on combined CIC-IDS2017 and CIC-IDS2018 datasets and evaluated on multiple independent test sets to measure generalization, robustness, and real-world applicability.

---

## Repository Structure

- Model_Trainer.py # Training script for RF classifier
- Model_Tester.py # Evaluation script for test files
- rf_model.joblib # Saved trained model (optional)
- rf_features.csv # Ordered features used by the model
- README.md # Project documentatione

---

## Features

### Training
- Loads and merges CIC-IDS2017 and CIC-IDS2018 CSV files.
- Auto-detects label column (label, class, attack, etc.).
- Converts all attacks into a single "Malicious" class.
- Cleans and normalizes numeric features.
- Optional SMOTE oversampling for class imbalance.
- Outputs:
  - Trained Random Forest model (`.joblib`)
  - Feature list CSV matching the training matrix
  - Evaluation metrics (precision, recall, F1, ROC-AUC)

### Evaluation
- Loads trained model and feature list.
- Accepts any CICFlowMeter-formatted test CSV.
- Automatically translates differing 2017/2018 column names.
- Reconstructs feature matrix exactly as during training.
- Produces:
  - Classification report
  - Confusion matrix
  - ROC curve (AUC)
  - Precision–Recall curve
  - Prediction summary (Benign vs. Malicious)

---

## Installation

### Requirements
Python 3.10+

### Install dependencies
pip install -r requirements.txt

---

## Training the Model

Example usage:

python Model_Trainer.py --data2017 path/to/CIC-IDS2017 --data2018 path/to/CIC-IDS2018 --modelfile rf_model.joblib --featuresfile rf_features.csv --metricsfile rf_metrics.csv --use_smote

Outputs include:
- `rf_model.joblib`
- `rf_features.csv`
- Training metrics (Precision, Recall, F1, ROC-AUC)

---

## Evaluating the Model

Example usage:

python Model_Tester.py --modelfile rf_model.joblib --featuresfile rf_features.csv --testfile Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv --year 2018

Outputs:
- Classification report
- Confusion matrix heatmap
- ROC and Precision–Recall curves
- Benign vs. Malicious prediction counts

---

## Datasets

This project uses the publicly available CIC-IDS2017 and CIC-IDS2018 datasets, which contain encrypted and unencrypted traffic, labeled attacks, and flow-based CICFlowMeter features.

## Results Summary

The Random Forest classifier performed well at detecting benign flows across all test sets. Malicious detection varied significantly depending on dataset characteristics such as attack rarity, imbalance, and distribution shift.

Key findings:
- High Benign F1 across all test files.
- Malicious Precision and Recall were inconsistent due to class imbalance.
- Some datasets contained extremely small malicious classes, causing the model to predict all benign.
- Strongest malicious detection occurred in datasets with larger attack samples.
- Performance degraded in datasets involving rare attack types or unseen patterns (e.g., port scans).

---

## Future Work

- Expand datasets beyond CIC-IDS (UGR’16, UNSW-NB15, MAWI, enterprise capture data).
- Standardize normalization across all datasets to enable true cross-dataset evaluation.
- Investigate feature reduction and selection techniques.
- Evaluate ensemble and gradient-boosted models (XGBoost, LightGBM).
- Explore deep learning approaches for encrypted traffic detection.
- Incorporate online learning and continuous model retraining pipelines.

---

## License
This project is intended for academic and research purposes. Refer to dataset owners for licensing details.


