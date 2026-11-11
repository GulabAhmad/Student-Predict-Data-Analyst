# 🎓 Student Pass/Fail Outcome Prediction System
**AI Engineering Bootcamp | Module 03: End-to-End Tabular Machine Learning Baseline**

---

## 📌 Executive Summary
This project delivers an end-to-end, reproducible tabular machine learning system to predict student **Pass/Fail** outcomes (`passed`: 0 = Fail, 1 = Pass) based on early behavioral indicators known **before** final examinations.

To strictly avoid data leakage, `final_score` (the final exam grade) is excluded from model training. The resulting pipeline achieves **82.14% Accuracy** and an **80.00% F1-Score** on a stratified held-out test set, significantly outperforming the baseline `DummyClassifier` (55.36% accuracy, 0.00% F1).

---

## 📂 Professional Project Structure
```text
PSEB Machine Learning Project/
│
├── data/                                  # All project datasets
│   ├── student_learning.csv               # Primary training dataset (280 records)
│   ├── new_student_records.csv            # Unseen student records for batch inference
│   ├── new_student_predictions_output.csv  # Output predictions with probabilities
│   └── test_set_error_analysis.csv        # Detailed test set misclassification audit
│
├── models/                                # Saved fitted models & pipelines
│   └── student_model_pipeline.joblib      # Persisted fitted pipeline (Preprocessor + Model)
│
├── notebooks/                             # Jupyter notebooks for interactive analysis
│   └── student_pass_fail_pipeline.ipynb   # Runnable notebook with graphs & evaluations
│
├── src/                                   # Source code (Core ML Pipeline)
│   └── student_pass_fail_pipeline.py      # Master standalone script (Steps 1–5)
│
├── docs/                                  # Project documentation & guidelines
│   ├── MODEL_CARD.md                      # Industry-standard model card
│   ├── EVALUATION_SUMMARY.md              # Benchmarks, confusion matrix & error analysis
│   └── RESPONSIBLE_AI.md                  # Ethical considerations, limitations & governance
│
├── app.py                                 # Interactive Streamlit Web Application (Step 6)
├── requirements.txt                       # Project Python dependencies
├── .gitignore                             # Git exclusion rules
├── README.md                              # Main project documentation
└── StudentName_TrainingID.zip             # Complete submission archive
```

---

## ⚙️ Setup & Installation

### 1. Open Terminal & Activate Environment
```bash
.venv\Scripts\activate
```
*(Or install dependencies if setting up on a new machine)*
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### Option 1: Run the Complete Pipeline (Terminal)
To execute the complete end-to-end ML workflow (Data loading, leakage-aware split, ColumnTransformer, baseline, model training, evaluation, saving pipeline, and predicting new records without retraining):
```bash
python src/student_pass_fail_pipeline.py
```
*(Or directly: `.venv\Scripts\python.exe src/student_pass_fail_pipeline.py`)*

### Option 2: Launch the Streamlit Web Application (Browser)
To launch the interactive GUI with real-time outcome prediction, probability sliders, tailored academic recommendations, and batch CSV predictions:
```bash
streamlit run app.py
```
*(Or directly: `.venv\Scripts\streamlit.exe run app.py`)*  
*The app will open automatically at `http://localhost:8501`.*

### Option 3: Run the Jupyter Notebook
Open `notebooks/student_pass_fail_pipeline.ipynb` in VS Code or JupyterLab and click **"Run All"**.

---

## 📊 Summary of Model Performance (Held-Out Test Set, N=56)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`DummyClassifier` - Most Frequent)** | 55.36% | 0.00% | 0.00% | 0.00% | 0.5000 |
| **Logistic Regression** | 75.00% | 72.00% | 72.00% | 72.00% | 0.8748 |
| **Random Forest Classifier (Selected Pipeline)** | **82.14%** | **80.00%** | **80.00%** | **80.00%** | **0.8452** |

---

## 🛡️ Responsible AI & Ethical Disclaimer
> **Important Rubric Notice:**  
> *This is synthetic classroom data. Predictions are strictly for demonstration and formative support. They must never be used to judge, punish, penalize, or deny educational opportunities to real students.*
