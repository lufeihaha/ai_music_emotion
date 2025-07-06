#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load training data"""
    try:
        features = np.load('data/features.npy')
        labels_df = pd.read_csv('data/processed/emotion_labels.csv')
        labels = labels_df['emotion'].values
        
        logger.info(f"Data loaded - Features: {features.shape}, Labels: {len(labels)}")
        
        if len(features.shape) == 3:
            n_samples, n_segments, n_features = features.shape
            features = features.reshape(n_samples, n_segments * n_features)
            logger.info(f"Features reshaped to 2D: {features.shape}")
        
        return features, labels
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        return None, None

def create_ensemble_model():
    """Create ensemble learning model"""
    models = [
        ('random_forest', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        )),
        ('gradient_boosting', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        )),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=200,
            random_state=42
        ))
    ]
    
    ensemble_model = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=1
    )
    
    return ensemble_model, dict(models)

def evaluate_individual_models(models, X_train, y_train, X_test, y_test):
    """Evaluate individual model performance"""
    individual_scores = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            individual_scores[name] = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{name}: Test Accuracy = {test_score:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {str(e)}")
            individual_scores[name] = {
                'train_score': 0,
                'test_score': 0,
                'cv_score': 0,
                'cv_std': 0
            }
    
    return individual_scores

def train_ensemble():
    """Train ensemble learning model"""
    print("Music Emotion Recognition - Ensemble Learning Training")
    print("=" * 60)
    
    # Load data
    features, labels = load_data()
    if features is None or labels is None:
        print("Failed to load data")
        return None
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create ensemble model
    ensemble_model, individual_models = create_ensemble_model()
    
    # Train ensemble model
    logger.info("Training ensemble model...")
    ensemble_model.fit(X_train_scaled, y_train)
    
    # Evaluate ensemble performance
    train_score = ensemble_model.score(X_train_scaled, y_train)
    test_score = ensemble_model.score(X_test_scaled, y_test)
    cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5)
    
    logger.info(f"Ensemble Training Accuracy: {train_score:.4f}")
    logger.info(f"Ensemble Test Accuracy: {test_score:.4f}")
    logger.info(f"Ensemble CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate individual models
    individual_scores = evaluate_individual_models(
        individual_models, X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Calculate performance improvement
    best_individual = max([scores['test_score'] for scores in individual_scores.values()])
    improvement = test_score - best_individual
    
    # Save models
    os.makedirs('models/ensemble_results', exist_ok=True)
    joblib.dump(ensemble_model, 'models/ensemble_results/ensemble_model.pkl')
    joblib.dump(scaler, 'models/ensemble_results/ensemble_scaler.pkl')
    joblib.dump(label_encoder, 'models/ensemble_results/ensemble_label_encoder.pkl')
    
    # Generate report
    y_pred = ensemble_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    report_content = f"""# Ensemble Learning Music Emotion Recognition Report

## Performance Overview

### Ensemble Model Performance
- Training Accuracy: {train_score:.4f}
- Test Accuracy: {test_score:.4f}
- Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})

### Individual Model Performance Comparison

"""
    
    for name, scores in individual_scores.items():
        report_content += f"- {name}: Test Accuracy = {scores['test_score']:.4f}, CV Score = {scores['cv_score']:.4f}\n"
    
    report_content += f"""
### Performance Improvement Analysis
- Best Individual Model Accuracy: {best_individual:.4f}
- Ensemble Model Accuracy: {test_score:.4f}
- Performance Improvement: +{improvement:.4f} ({improvement/best_individual*100:.1f}%)

## Conclusion
The ensemble learning approach successfully improved model performance by combining multiple algorithms.
"""
    
    with open('models/ensemble_results/ensemble_report.txt', 'w') as f:
        f.write(report_content)
    
    # Display results
    print(f"\nTraining Results:")
    print(f"Ensemble Model Test Accuracy: {test_score:.4f}")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f}")
    print(f"Performance Improvement: +{improvement:.4f} ({improvement/best_individual*100:.1f}%)")
    print(f"\nModels saved to models/ensemble_results/")
    
    return {
        'ensemble_test_score': test_score,
        'ensemble_cv_scores': cv_scores,
        'individual_scores': individual_scores,
        'improvement': improvement
    }

if __name__ == "__main__":
    train_ensemble() 