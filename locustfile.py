# locustfile.py
from locust import HttpUser, task, between

class IrisApiUser(HttpUser):
    wait_time = between(1, 2)  # Wait 1-2 seconds between requests

    @task
    def predict_endpoint(self):
        payload = {
          "sepal_length": 5.1,
          "sepal_width": 3.5,
          "petal_length": 1.4,
          "petal_width": 0.2
        }
        self.client.post("/predict", json=payload)