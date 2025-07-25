# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def train_model():
    """Trains a model and saves it."""
    print("Training model...")
    # Load the dataset
    df = pd.read_csv('iris.csv')
    
    # Define features (X) and target (y)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'species'
    
    X = df[features]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Initialize and train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy: {model.score(X_test, y_test):.4f}")
    
    # Save the model to a file
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("Model saved as model.pkl")

if __name__ == '__main__':
    train_model()