import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_extremely_large_values():
    sample_data = {
        "bill_length_mm": 9999.9,
        "bill_depth_mm": 9999.9,
        "flipper_length_mm": 9999,
        "body_mass_g": 999999
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code in (400, 422)

def test_empty_request():
    response = client.post("/predict", json={})
    assert response.status_code in (400, 422)
