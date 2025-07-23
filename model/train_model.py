import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

def train_iris_model():
    """Train and save the Iris classification model"""
    
    # Load the dataset
    df = pd.read_csv('iris.csv')
    
    # Prepare features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save the model
    joblib.dump(model, 'model/iris_model.pkl')
    
    # Save metrics for CML reporting
    metrics = {
        'accuracy': float(accuracy),
        'test_samples': int(len(y_test))
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    with open('classification_report.txt', 'w') as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Model trained successfully with accuracy: {accuracy:.4f}")
    return model

if __name__ == "__main__":
    train_iris_model()
