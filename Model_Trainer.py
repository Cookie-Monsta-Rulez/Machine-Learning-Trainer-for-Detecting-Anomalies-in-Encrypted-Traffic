#-----------------------------------
# Sean Cooke
# Professor Wagner
# CIS 735 - Machine Learning for Security
# Syracuse University
# Final Project
# December 10, 2025
# -----------------------------------

import argparse
import glob
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------
# Detect label column flexibly
# ---------------------------------------------------------
def find_label_column(df):
    cleaned_columns = {c.strip().lower(): c for c in df.columns}
    for key, original in cleaned_columns.items():
        if key in ["label", "class", "attack", "category"]:
            return original
    return None

# ---------------------------------------------------------
# Load all CSVs from a folder with auto-detection
# ---------------------------------------------------------
def load_folder(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    if not all_files:
        raise RuntimeError(f"No CSV files found in folder: {folder}")

    dfs = []
    for f in all_files:
        print(f"[INFO] Loading {f}")
        try:
            df = pd.read_csv(f, encoding='latin1', low_memory=False)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
            continue

        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        label_col = find_label_column(df)
        if label_col is None:
            print(f"[WARN] No label column found in {f}. Skipping.")
            continue

        df = df.dropna(axis=1, how='all')
        df = df.dropna(subset=[label_col])
        df = df.rename(columns={label_col: "Label"})

        if df["Label"].nunique() < 2:
            print(f"[WARN] Not enough class diversity in {f}. Skipping.")
            continue

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid CSV files with labels were found.")
    print(f"[INFO] Loaded {len(dfs)} valid CSV files.")
    return pd.concat(dfs, ignore_index=True)

# ---------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------
def main(args):
    print("\n===============================")
    print("     LOADING 2017 DATA")
    print("===============================")
    df17 = load_folder(args.data2017)

    print("\n===============================")
    print("     LOADING 2018 DATA")
    print("===============================")
    df18 = load_folder(args.data2018)

    print("[INFO] Concatenating datasets")
    df = pd.concat([df17, df18], ignore_index=True)

    # Convert labels to binary: Attack / Benign
    df["Label"] = df["Label"].str.strip()
    df["Label"] = df["Label"].replace({"BENIGN": "Benign", "Benign": "Benign"})
    df["Label"] = df["Label"].apply(lambda x: "Attack" if x != "Benign" else "Benign")

    # Keep numeric features only
    numeric_df = df.select_dtypes(include="number")

    # Replace inf and -inf with NaN, fill with median
    numeric_df = numeric_df.replace([float('inf'), float('-inf')], pd.NA)
    numeric_df = numeric_df.fillna(numeric_df.median())

    y = df["Label"]

    print("[INFO] Splitting dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        numeric_df, y, test_size=0.25, random_state=42, stratify=y
    )

    # Apply SMOTE if requested
    if args.use_smote:
        print("[INFO] Applying SMOTE to balance classes")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
    else:
        print("[INFO] Using class_weight='balanced' without SMOTE")
        X_train_res, y_train_res = X_train, y_train

    print("[INFO] Training Random Forest")
    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight='balanced' if not args.use_smote else None,
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # Save model
    model_file = args.modelfile or "rf_model_combined.joblib"
    joblib.dump(model, model_file)
    print(f"[INFO] Model saved to {model_file}")

    # Save feature list
    feature_file = args.featuresfile or "rf_features_combined.csv"
    pd.Series(X_train.columns).to_csv(feature_file, index=False, header=False)
    print(f"[INFO] Feature list saved to {feature_file}")

    # Align test features
    X_test = X_test[X_train.columns]

    print("\n[RESULTS] Evaluation:")
    preds = model.predict(X_test)
    report_dict = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))

    # ROC-AUC and PR-AUC
    y_true_bin = y_test.map({"Benign": 0, "Attack": 1})
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_true_bin, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"ROC-AUC Score: {roc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")

    # Save evaluation metrics to CSV
    metrics_file = args.metricsfile or "evaluation_metrics.csv"
    eval_df = pd.DataFrame(report_dict).transpose()
    eval_df.loc["ROC-AUC", "f1-score"] = roc
    eval_df.loc["PR-AUC", "f1-score"] = pr_auc
    eval_df.to_csv(metrics_file, index=True)
    print(f"[INFO] Evaluation metrics saved to {metrics_file}")

# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data2017", required=True, help="Path to 2017 CSV folder")
    parser.add_argument("--data2018", required=True, help="Path to 2018 CSV folder")
    parser.add_argument("--modelfile", default=None, help="Filename to save the trained model")
    parser.add_argument("--featuresfile", default=None, help="Filename to save the feature list")
    parser.add_argument("--metricsfile", default=None, help="Filename to save evaluation metrics")
    parser.add_argument("--use_smote", action='store_true', help="Apply SMOTE to balance classes")
    args = parser.parse_args()
    main(args)
