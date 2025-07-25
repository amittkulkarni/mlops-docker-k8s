# Project File Descriptions

### `train.py`
Trains a machine learning model on the Iris dataset and saves the trained model as `model.pkl`.

### `main.py`
A FastAPI application that loads the trained model and exposes a `/predict` endpoint to serve predictions over the internet.

### `requirements.txt`
Lists all the Python libraries and dependencies that are required to run the project.

### `Dockerfile`
Contains the instructions to build a Docker container image for the application. It installs dependencies, copies the code, and runs the training script.

### `iris.csv`
The dataset containing feature measurements for three species of Iris flowers, used to train the machine learning model.

### `kubernetes/deployment.yaml`
A Kubernetes manifest that defines the desired state for the application on GKE. It specifies the container image, number of replicas, and labels.

### `kubernetes/service.yaml`
A Kubernetes manifest that creates a `LoadBalancer`. This exposes the application to the internet with a stable, public IP address.

### `.github/workflows/cd-pipeline.yaml`
The CI/CD pipeline script for GitHub Actions. It automates building the Docker image, pushing it to Artifact Registry, and deploying the application to GKE.