"""
========================================================================================
END-TO-END TABULAR MACHINE LEARNING PIPELINE: STUDENT PASS/FAIL PREDICTION
Module 03 | Beginner ML Project (Complete All-in-One Python Script)
========================================================================================
This single file executes the entire rubric workflow:
1. Data Inspection & Feature Selection (Leakage-aware: final_score excluded)
2. Stratified Train/Test Split & Dummy Baseline Model
3. Preprocessing with ColumnTransformer & Pipeline (Imputation, Scaling, One-Hot Encoding)
4. Model Training & Fair Evaluation (Dummy Baseline, Logistic Regression, Random Forest)
5. Model Persistence with Joblib & Direct Prediction on new_student_records.csv
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -----------------------------------------------------------------------------
# Dynamic Path Resolution (Works whether executed from root or src/)
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "student_learning.csv")
if not os.path.exists(CSV_FILE):
    # Fallback to local
    CSV_FILE = "student_learning.csv"

def main():
    print("=" * 80)
    print("STEP 1: DATA INSPECTION & FEATURE SELECTION")
    print("=" * 80)
    
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Dataset '{CSV_FILE}' not found.")
        
    df = pd.read_csv(CSV_FILE)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("First 5 rows:")
    print(df.head())
    
    print("\nColumn Data Types:")
    print(df.dtypes)
    
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    
    print("\nTarget Balance ('passed'):")
    print(df["passed"].value_counts(dropna=False))
    print(df["passed"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    
    # Feature Selection & Leakage Prevention:
    # Exclude 'final_score' because it represents post-exam final grade and causes target leakage.
    target_col = "passed"
    leakage_col = "final_score"
    
    X = df.drop(columns=[target_col, leakage_col])
    y = df[target_col]
    
    numerical_cols = ["hours_studied", "attendance", "practice_tests", "sleep_hours"]
    categorical_cols = ["study_method"]
    
    print(f"\nSelected Input Features ({len(X.columns)}):", list(X.columns))
    print(f"Target Column: '{target_col}'")
    print(f"Excluded Leakage Column: '{leakage_col}'")

    print("\n" + "=" * 80)
    print("STEP 2: STRATIFIED TRAIN/TEST SPLIT & DUMMY BASELINE")
    print("=" * 80)
    
    # Reproducible stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training instances: {X_train.shape[0]} (Fail: {(y_train == 0).sum()}, Pass: {(y_train == 1).sum()})")
    print(f"Testing instances:  {X_test.shape[0]} (Fail: {(y_test == 0).sum()}, Pass: {(y_test == 1).sum()})")

    # Baseline Model: DummyClassifier (most frequent strategy)
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    
    baseline_acc = accuracy_score(y_test, baseline_preds)
    baseline_f1 = f1_score(y_test, baseline_preds, zero_division=0)
    print(f"\nBaseline (Dummy) Accuracy: {baseline_acc:.4f}")
    print(f"Baseline (Dummy) F1-Score: {baseline_f1:.4f}")
    print("Baseline Confusion Matrix:\n", confusion_matrix(y_test, baseline_preds))

    print("\n" + "=" * 80)
    print("STEP 3: PREPROCESSING PIPELINE (ColumnTransformer)")
    print("=" * 80)
    
    # Numerical: Median imputer + Standard scaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical: Most frequent imputer + One-Hot Encoder
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])
    print("ColumnTransformer constructed successfully.")

    print("\n" + "=" * 80)
    print("STEP 4: TRAIN CANDIDATE MODELS & FAIR EVALUATION")
    print("=" * 80)
    
    candidates = {
        "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    fitted_pipelines = {}
    eval_rows = []
    
    for name, clf in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        preds = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, prob) if prob is not None else 0.5
        cm = confusion_matrix(y_test, preds)
        
        eval_rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Confusion Matrix": cm
        })
    
    summary_df = pd.DataFrame(eval_rows)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 85)
    for row in eval_rows:
        print(f"{row['Model']:<25} | {row['Accuracy']:<10.4f} | {row['Precision']:<10.4f} | {row['Recall']:<10.4f} | {row['F1-Score']:<10.4f} | {row['ROC-AUC']:<10.4f}")

    best_model_name = "Random Forest"
    best_pipe = fitted_pipelines[best_model_name]
    best_preds = best_pipe.predict(X_test)
    
    print(f"\n--- Detailed Classification Report for Best Model ({best_model_name}) ---")
    print(classification_report(y_test, best_preds, target_names=["Fail (0)", "Pass (1)"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, best_preds))

    # Error analysis
    test_eval = X_test.copy()
    test_eval["actual"] = y_test
    test_eval["predicted"] = best_preds
    test_eval["prob"] = np.round(best_pipe.predict_proba(X_test)[:, 1], 3)
    errors = test_eval[test_eval["actual"] != test_eval["predicted"]]
    print(f"\nTotal Errors on Held-Out Test Set: {len(errors)} / {len(X_test)}")
    print(f"False Positives (predicted pass, actually failed): {len(test_eval[(test_eval['actual']==0) & (test_eval['predicted']==1)])}")
    print(f"False Negatives (predicted fail, actually passed): {len(test_eval[(test_eval['actual']==1) & (test_eval['predicted']==0)])}")

    print("\n" + "=" * 80)
    print("STEP 5: SAVE PIPELINE & PREDICT NEW RECORDS WITHOUT RETRAINING")
    print("=" * 80)
    
    # Save the fitted pipeline in models/ directory
    model_filepath = os.path.join(MODELS_DIR, "student_model_pipeline.joblib")
    joblib.dump(best_pipe, model_filepath)
    print(f"Saved complete fitted pipeline to '{model_filepath}'")
    
    # Create sample new_student_records.csv if not exists in data/
    new_csv = os.path.join(DATA_DIR, "new_student_records.csv")
    if not os.path.exists(new_csv):
        new_data = pd.DataFrame([
            {"hours_studied": 8.5, "attendance": 90, "practice_tests": 4, "sleep_hours": 7.5, "study_method": "group"},
            {"hours_studied": 2.1, "attendance": 60, "practice_tests": 1, "sleep_hours": 5.0, "study_method": "self"},
            {"hours_studied": 5.5, "attendance": 78, "practice_tests": 3, "sleep_hours": 6.8, "study_method": "video"},
            {"hours_studied": 7.0, "attendance": 85, "practice_tests": 5, "sleep_hours": 8.0, "study_method": "self"},
            {"hours_studied": 1.5, "attendance": 55, "practice_tests": 0, "sleep_hours": 6.0, "study_method": "video"},
            {"hours_studied": 4.8, "attendance": 70, "practice_tests": 2, "sleep_hours": 7.2, "study_method": "group"},
        ])
        new_data.to_csv(new_csv, index=False)
        print(f"Created sample '{new_csv}'")
    
    # Reload pipeline from disk and predict directly without retraining
    print(f"\nReloading pipeline from '{model_filepath}'...")
    reloaded_pipeline = joblib.load(model_filepath)
    
    new_records_df = pd.read_csv(new_csv)
    predictions = reloaded_pipeline.predict(new_records_df)
    probabilities = reloaded_pipeline.predict_proba(new_records_df)[:, 1]
    
    results_df = new_records_df.copy()
    results_df["predicted_passed"] = predictions
    results_df["outcome"] = np.where(predictions == 1, "Pass", "Fail")
    results_df["pass_probability"] = np.round(probabilities, 4)
    
    print("\nPredictions on new records:")
    print(results_df)
    output_csv = os.path.join(DATA_DIR, "new_student_predictions_output.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions output to '{output_csv}'")
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
