# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

valid_input = {
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "year": 2007,
    "sex": "male",
    "island": "Torgersen"
}

def test_predict_endpoint_valid_input():
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    json_resp = response.json()
    assert "predicted_species" in json_resp
    assert isinstance(json_resp["predicted_species"], str)

@pytest.mark.parametrize("missing_field", [
    "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year", "sex", "island"
])
def test_missing_fields(missing_field):
    data = valid_input.copy()
    data.pop(missing_field)
    response = client.post("/predict", json=data)
    assert response.status_code == 422  # validation error from FastAPI/Pydantic

@pytest.mark.parametrize("field,value", [
    ("bill_length_mm", "invalid"),  # string instead of float
    ("body_mass_g", -100),          # out-of-range negative value
    ("sex", "unknown"),             # invalid enum value
    ("island", "unknown")           # invalid enum value
])
def test_invalid_inputs(field, value):
    data = valid_input.copy()
    data[field] = value
    response = client.post("/predict", json=data)
    # Could be 422 (validation) or 400 (custom), so accept 422 or 400 here
    assert response.status_code in (400, 422)

def test_edge_case_extreme_values():
    extreme_input = valid_input.copy()
    extreme_input["bill_length_mm"] = 9999.9
    extreme_input["body_mass_g"] = 0  # edge: zero mass
    response = client.post("/predict", json=extreme_input)
    # Prediction should still work, may or may not be realistic but no crash
    assert response.status_code == 200
    assert "predicted_species" in response.json()
