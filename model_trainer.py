import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X, y, model_dir="models"):
    """Train a model to predict lineup effectiveness"""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Try a different model - Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=15,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_predictions = model.predict(X_test)
    test_score = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    print(f"Model R² on training data: {train_score:.4f}")
    print(f"Model R² on test data: {test_score:.4f}")
    print(f"Model RMSE on test data: {test_rmse:.4f}")
    
    return model

def save_model(model, encoders, model_dir="models", filename="fifth_player_predictor.pkl"):
    """Save model and encoders to a file"""
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump((model, encoders), f)
    
    print(f"Model saved to {filepath}")
    
    return filepath

def analyze_feature_importance(model, feature_names):
    """Analyze which features are most important for predictions"""
    # Get feature importances from model
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    imp_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    })
    
    # Sort by importance
    imp_df = imp_df.sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for i, row in imp_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    return imp_df