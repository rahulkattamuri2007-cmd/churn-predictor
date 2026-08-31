import joblib
from sklearn.ensemble import RandomForestClassifier

def test_saved_model():
    print("Loading model.pkl and model_columns.pkl...")
    
    # 1. Load model and feature columns
    model = joblib.load('model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    
    # 2. Verify model object type
    print("\n=== Model Verification ===")
    print(f"Loaded model type: {type(model)}")
    is_rf = isinstance(model, RandomForestClassifier)
    print(f"Is valid RandomForestClassifier? {is_rf}")
    assert is_rf, "Loaded model is not a RandomForestClassifier!"
    
    # 3. Verify model parameters & columns
    print("\n=== Columns Verification ===")
    print(f"Loaded columns type: {type(model_columns)}")
    print(f"Total feature columns: {len(model_columns)}")
    print(f"First 5 feature names: {model_columns[:5]}")
    assert isinstance(model_columns, list), "Loaded columns object is not a list!"
    assert len(model_columns) == 30, f"Expected 30 columns, got {len(model_columns)}"
    
    print("\n[SUCCESS] Both model.pkl and model_columns.pkl loaded successfully and verified!")

if __name__ == '__main__':
    test_saved_model()
