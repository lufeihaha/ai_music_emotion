#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

def quick_ensemble_test():
    """Quick test of ensemble learning approach"""
    print("Quick Ensemble Learning Test")
    print("=" * 40)
    
    try:
        # Load data
        features = np.load('data/features.npy')
        labels_df = pd.read_csv('data/processed/emotion_labels.csv')
        labels = labels_df['emotion'].values
        
        print(f"Data loaded - Features: {features.shape}, Labels: {len(labels)}")
        
        # Handle 3D features
        if len(features.shape) == 3:
            n_samples, n_segments, n_features = features.shape
            features = features.reshape(n_samples, n_segments * n_features)
            print(f"Features reshaped to: {features.shape}")
        
        # Take a smaller sample for quick testing
        sample_size = min(200, len(features))
        indices = np.random.choice(len(features), sample_size, replace=False)
        features_sample = features[indices]
        labels_sample = labels[indices]
        
        print(f"Using sample size: {sample_size}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels_sample)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_sample, labels_encoded, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        # Define simple models
        models = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
        ]
        
        # Test individual models
        individual_scores = {}
        for name, model in models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            individual_scores[name] = score
            print(f"{name}: {score:.4f}")
        
        # Test ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        ensemble_score = accuracy_score(y_test, y_pred_ensemble)
        
        print(f"Ensemble: {ensemble_score:.4f}")
        
        # Calculate improvement
        best_individual = max(individual_scores.values())
        improvement = ensemble_score - best_individual
        
        print(f"\nResults:")
        print(f"Best individual model: {best_individual:.4f}")
        print(f"Ensemble model: {ensemble_score:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement/best_individual*100:.1f}%)")
        
        return ensemble_score > best_individual
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = quick_ensemble_test()
    if success:
        print("\n✅ Ensemble learning shows improvement!")
    else:
        print("\n❌ Ensemble learning needs further optimization") 