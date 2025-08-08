from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_valid_input():
    """Test prediction endpoint with valid penguin data"""
    sample_data = {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
