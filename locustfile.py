from locust import HttpUser, task, between

class PenguinApiUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task
    def predict(self):
        sample_data = {
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181,
            "body_mass_g": 3750
        }
        self.client.post("/predict", json=sample_data)