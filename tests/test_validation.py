import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_missing_fields():
    sample_data = {
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code in (400, 422)

def test_invalid_data_types():
    sample_data = {
        "bill_length_mm": "hello",
        "bill_depth_mm": "world",
        "flipper_length_mm": "foo",
        "body_mass_g": "bar"
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code in (400, 422)

def test_negative_values():
    sample_data = {
        "bill_length_mm": -5,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": -200
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code in (400, 422)
