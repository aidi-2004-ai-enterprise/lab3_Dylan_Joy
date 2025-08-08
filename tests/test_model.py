import os
import xgboost as xgb
import pytest

MODEL_PATH = os.path.join("app", "data", "model.json")

@pytest.fixture(scope="module")
def model():
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    return booster

def test_model_prediction(model):
    """Test model can make predictions on known data"""
    dmatrix = xgb.DMatrix([[39.1, 18.7, 181, 3750]])
    prediction = model.predict(dmatrix)
    assert prediction.shape == (1,)
