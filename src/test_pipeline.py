"""
test_pipeline.py
Automated Verification & Unit Tests for the Student Pass/Fail Prediction Pipeline.
"""
import os
import joblib
import pandas as pd
import numpy as np

def test_model_loading():
    """Verify that the persisted model file exists and loads cleanly."""
    model_path = os.path.join("models", "student_model_pipeline.joblib")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    pipeline = joblib.load(model_path)
    assert pipeline is not None, "Pipeline failed to load."
    assert hasattr(pipeline, "predict"), "Pipeline missing predict method."
    print("✓ Model loading test passed.")

def test_prediction_output_shape():
    """Verify that predictions match input row count and valid classes."""
    model_path = os.path.join("models", "student_model_pipeline.joblib")
    pipeline = joblib.load(model_path)
    
    sample_input = pd.DataFrame([{
        "hours_studied": 6.0,
        "attendance": 80,
        "practice_tests": 3,
        "sleep_hours": 7.0,
        "study_method": "group"
    }])
    
    pred = pipeline.predict(sample_input)
    prob = pipeline.predict_proba(sample_input)
    
    assert len(pred) == 1, "Prediction output length mismatch."
    assert pred[0] in [0, 1], "Prediction class must be 0 or 1."
    assert prob.shape == (1, 2), "Probability output shape mismatch."
    assert np.isclose(prob[0].sum(), 1.0), "Probabilities must sum to 1."
    print("✓ Single-sample prediction shape & probability test passed.")

def test_batch_inference():
    """Verify that batch inference works on new_student_records.csv."""
    model_path = os.path.join("models", "student_model_pipeline.joblib")
    data_path = os.path.join("data", "new_student_records.csv")
    pipeline = joblib.load(model_path)
    
    data = pd.read_csv(data_path)
    preds = pipeline.predict(data)
    assert len(preds) == len(data), "Batch prediction count mismatch."
    print(f"✓ Batch inference test passed on {len(data)} records.")

if __name__ == "__main__":
    print("Running automated pipeline verification tests...")
    test_model_loading()
    test_prediction_output_shape()
    test_batch_inference()
    print("All tests passed successfully!")

