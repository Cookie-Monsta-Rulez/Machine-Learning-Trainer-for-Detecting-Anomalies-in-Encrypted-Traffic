#-----------------------------------
# Sean Cooke
# Professor Wagner
# CIS 735 - Machine Learning for Security
# Syracuse University
# Final Project
# December 10, 2025
# -----------------------------------
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
from tqdm import tqdm
import argparse
import os

# ---------------------------
# COMMAND-LINE ARGUMENTS
# ---------------------------
parser = argparse.ArgumentParser(description="Evaluate a ML model on test data.")
parser.add_argument("--modelfile", type=str, required=True, help="Path to trained model (.joblib)")
parser.add_argument("--featuresfile", type=str, required=True, help="CSV containing feature list")
parser.add_argument("--testfile", type=str, required=True, help="CSV containing test dataset")
parser.add_argument("--year", type=int, choices=[2017, 2018], default=2018, help="Year of CICFlowMeter file format")
args = parser.parse_args()

MODEL_FILE = args.modelfile
FEATURES_FILE = args.featuresfile
TEST_FILE = args.testfile
YEAR = args.year

# ---------------------------
# COLUMN NAME TRANSLATION MAP
# ---------------------------
if YEAR == 2017:
    column_map = {
        "Destination Port": "Destination Port",
        "Src Port": "Src Port",
        "Total Fwd Packets": "Total Fwd Packets",
        "Total Backward Packets": "Total Backward Packets",
        "Fwd Header Length": "Fwd Header Length",
        "Bwd Header Length": "Bwd Header Length",
        # ... add the rest of 2017 mapping
    }
elif YEAR == 2018:
    column_map = {
        "Destination Port": "Dst Port",
        "Src Port": "Src Port",
        "Total Fwd Packets": "Tot Fwd Pkts",
        "Total Backward Packets": "Tot Bwd Pkts",
        "Total Length of Fwd Packets": "TotLen Fwd Pkts",
        "Total Length of Bwd Packets": "TotLen Bwd Pkts",
        "Fwd Packet Length Max": "Fwd Pkt Len Max",
        "Fwd Packet Length Min": "Fwd Pkt Len Min",
        "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
        "Fwd Packet Length Std": "Fwd Pkt Len Std",
        "Bwd Packet Length Max": "Bwd Pkt Len Max",
        "Bwd Packet Length Min": "Bwd Pkt Len Min",
        "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
        "Bwd Packet Length Std": "Bwd Pkt Len Std",
        "Flow Bytes/s": "Flow Byts/s",
        "Flow Packets/s": "Flow Pkts/s",
        "Flow IAT Mean": "Flow IAT Mean",
        "Flow IAT Std": "Flow IAT Std",
        "Flow IAT Max": "Flow IAT Max",
        "Flow IAT Min": "Flow IAT Min",
        "Fwd IAT Total": "Fwd IAT Tot",
        "Fwd IAT Mean": "Fwd IAT Mean",
        "Fwd IAT Std": "Fwd IAT Std",
        "Fwd IAT Max": "Fwd IAT Max",
        "Fwd IAT Min": "Fwd IAT Min",
        "Bwd IAT Total": "Bwd IAT Tot",
        "Bwd IAT Mean": "Bwd IAT Mean",
        "Bwd IAT Std": "Bwd IAT Std",
        "Bwd IAT Max": "Bwd IAT Max",
        "Bwd IAT Min": "Bwd IAT Min",
        "Fwd PSH Flags": "Fwd PSH Flags",
        "Bwd PSH Flags": "Bwd PSH Flags",
        "Fwd URG Flags": "Fwd URG Flags",
        "Bwd URG Flags": "Bwd URG Flags",
        "Fwd Header Length": "Fwd Header Len",
        "Bwd Header Length": "Bwd Header Len",
        "Fwd Packets/s": "Fwd Pkts/s",
        "Bwd Packets/s": "Bwd Pkts/s",
        "Min Packet Length": "Pkt Len Min",
        "Max Packet Length": "Pkt Len Max",
        "Packet Length Mean": "Pkt Len Mean",
        "Packet Length Std": "Pkt Len Std",
        "Packet Length Variance": "Pkt Len Var",
        "FIN Flag Count": "FIN Flag Cnt",
        "SYN Flag Count": "SYN Flag Cnt",
        "RST Flag Count": "RST Flag Cnt",
        "PSH Flag Count": "PSH Flag Cnt",
        "ACK Flag Count": "ACK Flag Cnt",
        "URG Flag Count": "URG Flag Cnt",
        "CWE Flag Count": "CWE Flag Count",
        "ECE Flag Count": "ECE Flag Cnt",
        # ... add rest of mapping as needed
    }

# ---------------------------
# CHECK FILES EXIST
# ---------------------------
for fpath in [MODEL_FILE, FEATURES_FILE, TEST_FILE]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File not found: {fpath}")

# ---------------------------
# LOAD MODEL + FEATURE LIST
# ---------------------------
rf_model = joblib.load(MODEL_FILE)
feature_df = pd.read_csv(FEATURES_FILE, header=None)
rf_features = feature_df.iloc[:, 0].tolist()

# ---------------------------
# LOAD TEST DATA
# ---------------------------
df_test = pd.read_csv(TEST_FILE, engine="python", on_bad_lines="skip")
df_test = df_test.rename(columns=lambda x: x.strip())
print(f"[INFO] Loaded test set: {df_test.shape[0]} rows, {df_test.shape[1]} columns")

# ---------------------------
# EXTRACT LABEL
# ---------------------------
if "Label" in df_test.columns:
    y_true = df_test["Label"].astype(str).str.upper().apply(lambda x: 0 if "BENIGN" in x else 1)
else:
    y_true = None
    print("[WARN] No Label column present in test data.")

# ---------------------------
# BUILD X_test
# ---------------------------
X_test = pd.DataFrame()
missing_features = []
for f in tqdm(rf_features, desc="Building feature matrix"):
    col = column_map.get(f, f)
    if col in df_test.columns:
        X_test[f] = pd.to_numeric(df_test[col], errors="coerce")
    else:
        X_test[f] = np.nan
        missing_features.append(f)

# Fill missing values with median
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(X_test.median(), inplace=True)

if missing_features:
    print(f"[WARN] {len(missing_features)} mapped features missing in test set, filled with median: {missing_features}")

# Ensure order matches training
X_test = X_test[rf_features]
print(f"[INFO] Final X_test shape: {X_test.shape}")

# ---------------------------
# PREDICTIONS
# ---------------------------
y_pred_raw = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1] if hasattr(rf_model, "predict_proba") else y_pred_raw

if isinstance(y_pred_raw[0], str):
    y_pred = np.array([0 if "BENIGN" in str(x).upper() else 1 for x in y_pred_raw])
else:
    y_pred = y_pred_raw

# ---------------------------
# EVALUATION REPORT
# ---------------------------
print("\n=== EVALUATION REPORT ===\n")
if y_true is not None:
    print(classification_report(y_true, y_pred, target_names=["BENIGN", "MALICIOUS"], zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["BENIGN","MALICIOUS"], yticklabels=["BENIGN","MALICIOUS"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()
else:
    print("[WARN] True labels not available; skipping metrics and curves.")

print("[INFO] Prediction complete. Summary:")
print(f" - Total samples: {len(X_test)}")
print(f" - Predicted MALICIOUS: {np.sum(y_pred == 1)}")
print(f" - Predicted BENIGN: {np.sum(y_pred == 0)}")
