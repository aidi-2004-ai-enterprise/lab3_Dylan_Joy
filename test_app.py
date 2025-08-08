from fastapi.testclient import TestClient
from app.main import app  # correct import path

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Penguin species prediction API is running."
