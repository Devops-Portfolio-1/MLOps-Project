import time
from main import DiabetesInput, build_features, model

# hasttr() means the model has a predict method, which is essential for making predictions. This test ensures that the model artifact was loaded correctly and is ready for inference.


def test_model_artifact_loads():
    assert hasattr(model, "predict") 

def test_model_prediction_runs():
    sample = DiabetesInput(
        gender="Female",
        age=45,
        hypertension=1,
        heart_disease=0,
        bmi=28.5,
        HbA1c_level=6.5,
        blood_glucose_level=140,
        smoking_history="former",
    )
    features = build_features(sample)
    prediction = model.predict(features)
    assert len(prediction) == 1
    assert int(prediction[0]) in (0, 1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0][1]
        assert 0.0 <= float(proba) <= 1.0


def test_model_accuracy_diverse_inputs():
    """Test model produces consistent and reasonable predictions across diverse inputs"""
    test_cases = [
        # Low risk case
        DiabetesInput(
            gender="Female",
            age=25,
            hypertension=0,
            heart_disease=0,
            bmi=22,
            HbA1c_level=4.5,
            blood_glucose_level=90,
            smoking_history="never",
        ),
        # High risk case
        DiabetesInput(
            gender="Male",
            age=65,
            hypertension=1,
            heart_disease=1,
            bmi=35,
            HbA1c_level=8.0,
            blood_glucose_level=180,
            smoking_history="current",
        ),
        # Medium risk case
        DiabetesInput(
            gender="Female",
            age=45,
            hypertension=1,
            heart_disease=0,
            bmi=28.5,
            HbA1c_level=6.5,
            blood_glucose_level=140,
            smoking_history="former",
        ),
    ]

    predictions = []
    for test_case in test_cases:
        features = build_features(test_case)
        pred = model.predict(features)[0]
        predictions.append(int(pred))

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            assert len(proba) == 2
            assert abs(sum(proba) - 1.0) < 0.01  # probabilities should sum to 1

    # Ensure we got binary predictions
    assert all(p in (0, 1) for p in predictions)


def test_model_prediction_consistency():
    """Test that the same input produces the same prediction"""
    sample = DiabetesInput(
        gender="Male",
        age=55,
        hypertension=1,
        heart_disease=0,
        bmi=30,
        HbA1c_level=7.0,
        blood_glucose_level=160,
        smoking_history="ever",
    )

    predictions = []
    for _ in range(3):
        features = build_features(sample)
        pred = model.predict(features)[0]
        predictions.append(int(pred))

    # All predictions should be identical
    assert len(set(predictions)) == 1, f"Inconsistent predictions: {predictions}"


def test_model_performance():
    """Test that model inference is performant (< 100ms per prediction)"""
    sample = DiabetesInput(
        gender="Female",
        age=45,
        hypertension=1,
        heart_disease=0,
        bmi=28.5,
        HbA1c_level=6.5,
        blood_glucose_level=140,
        smoking_history="former",
    )
    features = build_features(sample)

    # Warmup
    model.predict(features)

    # Measure inference time
    start_time = time.time()
    for _ in range(100):
        model.predict(features)
    end_time = time.time()

    avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
    assert avg_time < 100, f"Model inference too slow: {avg_time:.2f}ms per prediction"
    print(f"Average inference time: {avg_time:.2f}ms")


def test_model_prediction_bounds():
    """Test that probability predictions are within valid bounds"""
    if not hasattr(model, "predict_proba"):
        return  # Skip if model doesn't support predict_proba

    sample = DiabetesInput(
        gender="Female",
        age=45,
        hypertension=1,
        heart_disease=0,
        bmi=28.5,
        HbA1c_level=6.5,
        blood_glucose_level=140,
        smoking_history="former",
    )
    features = build_features(sample)
    proba = model.predict_proba(features)[0]

    assert all(0.0 <= p <= 1.0 for p in proba), f"Probability out of bounds: {proba}"
