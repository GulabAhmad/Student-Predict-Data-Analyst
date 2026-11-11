# 📋 Model Card: Student Pass/Fail Prediction Pipeline

## 1. Model Details
- **Model Name:** Student Academic Outcome Predictor (Random Forest Pipeline)
- **Model Version:** 1.0.0
- **Model Type:** Supervised Binary Classification Pipeline
- **Frameworks:** Python 3.12, scikit-learn 1.9, Joblib 1.6
- **Pipeline Architecture:**
  - **Preprocessor:** `ColumnTransformer`
    - Numeric (`hours_studied`, `attendance`, `practice_tests`, `sleep_hours`): `SimpleImputer(strategy='median')` + `StandardScaler()`
    - Categorical (`study_method`): `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`
  - **Estimator:** `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)`

---

## 2. Intended Use
- **Primary Intended Use:** Formative early-warning system in higher education to identify at-risk students midway through a semester and proactively offer academic counseling, peer tutoring, and study support.
- **Out-of-Scope / Prohibited Uses:**
  - Automatic pass/fail assignment or formal grading.
  - Revoking scholarships, denying exam admission, or imposing disciplinary sanctions.
  - Making automated high-stakes decisions without human counselor review.

---

## 3. Factors & Features
- **Input Features Used (Pre-Exam Predictors):**
  1. `hours_studied`: Continuous (weekly hours dedicated to course revision).
  2. `attendance`: Integer percentage (0–100% of classes attended).
  3. `practice_tests`: Count (0–5 diagnostic tests completed).
  4. `sleep_hours`: Continuous (average hours of sleep per night).
  5. `study_method`: Categorical (`self`, `group`, `video`).
- **Excluded Features (Target Leakage):**
  - `final_score`: Excluded because it is the post-exam final grade which directly dictates `passed`.

---

## 4. Evaluation Data & Performance Metrics
- **Dataset:** `student_learning.csv` (280 records; 56% Fail / 44% Pass).
- **Split Protocol:** Stratified 80/20 train/test split with `random_state=42`. Test set contains 56 students (31 Fail, 25 Pass).
- **Metrics on Held-Out Test Set:**
  - **Accuracy:** `82.14%`
  - **F1-Score (Binary, Class 1):** `80.00%`
  - **Precision:** `80.00%`
  - **Recall:** `80.00%`
  - **ROC-AUC:** `0.8452`
  - **Baseline Dummy Comparison:** 55.36% Accuracy, 0.00% F1.

---

## 5. Confusion Matrix & Error Profile
```text
               Predicted Fail (0)   Predicted Pass (1)
Actual Fail (0)        26                    5      (False Positives)
Actual Pass (1)         5                   20      (True Positives)
```
- **False Positives (5):** Student predicted to pass but actually failed.
- **False Negatives (5):** Student predicted to fail but actually passed.
- **Error Characteristics:** Errors cluster near the decision boundary (students with 4.0–7.0 study hours) where unmeasured latent variables (such as exam difficulty, sickness, or test anxiety) decide the outcome.

---

## 6. Ethical Considerations & Limitations
1. **Synthetic Data Disclaimer:** This dataset is synthetic. Real classroom performance depends on diverse human and environmental factors not captured here.
2. **Missing Covariates:** The model does not account for socioeconomic disparities, access to quiet study environments, caregiving responsibilities, or disability accommodations.
3. **Actionability:** Predictions should always trigger constructive resources (free tutoring, flexible office hours) rather than negative labeling.

