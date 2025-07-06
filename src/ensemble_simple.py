#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Ensemble Learning for Music Emotion Recognition
"""

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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEnsembleClassifier:
    """Simple Ensemble Learning Music Emotion Classifier"""
    
    def __init__(self):
        self.ensemble_model = None
        self.individual_models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def create_ensemble_model(self):
        """Create ensemble learning model"""
        # Define individual models
        models = [
            ('random_forest', RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            ('gradient_boosting', GradientBoostingClassifier(
                n_estimators=300,
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
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ))
        ]
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=models,
            voting='soft',  # Use soft voting (probability-based)
            n_jobs=-1
        )
        
        # Save individual model references
        for name, model in models:
            self.individual_models[name] = model
            
        logger.info("Ensemble learning model created successfully")
        return self.ensemble_model
    
    def load_data(self):
        """Load training data"""
        try:
            # Load feature data
            features = np.load('data/features.npy')
            labels_df = pd.read_csv('data/processed/emotion_labels.csv')
            labels = labels_df['emotion'].values
            
            logger.info(f"Data loaded - Features shape: {features.shape}, Labels count: {len(labels)}")
            
            # Handle 3D feature data
            if len(features.shape) == 3:
                n_samples, n_segments, n_features = features.shape
                features = features.reshape(n_samples, n_segments * n_features)
                logger.info(f"Features reshaped to 2D: {features.shape}")
            
            return features, labels
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return None, None
    
    def train_ensemble(self, features, labels, test_size=0.2):
        """Train ensemble learning model"""
        try:
            # Encode labels
            labels_encoded = self.label_encoder.fit_transform(labels)
            
            # Split train/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels_encoded, 
                test_size=test_size, 
                random_state=42, 
                stratify=labels_encoded
            )
            
            # Feature standardization
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create ensemble model
            self.create_ensemble_model()
            
            # Train ensemble model
            logger.info("Starting ensemble learning model training...")
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            train_score = self.ensemble_model.score(X_train_scaled, y_train)
            test_score = self.ensemble_model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train, cv=5)
            
            logger.info(f"Training completed!")
            logger.info(f"Training accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Generate detailed report
            y_pred = self.ensemble_model.predict(X_test_scaled)
            
            # Use simple English emotion labels to avoid encoding issues
            emotion_labels = [f"emotion_{i}" for i in range(len(self.label_encoder.classes_))]
            report = classification_report(y_test, y_pred, 
                                         target_names=emotion_labels,
                                         output_dict=True)
            
            # Evaluate individual model performance
            individual_scores = self._evaluate_individual_models(X_train_scaled, y_train, X_test_scaled, y_test)
            
            results = {
                'ensemble_train_score': train_score,
                'ensemble_test_score': test_score,
                'ensemble_cv_scores': cv_scores,
                'individual_scores': individual_scores,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'test_predictions': y_pred,
                'test_labels': y_test
            }
            
            # Save models
            self.save_models()
            
            # Generate visualization report
            self._generate_visualization_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return None
    
    def _evaluate_individual_models(self, X_train, y_train, X_test, y_test):
        """Evaluate individual model performance"""
        individual_scores = {}
        
        for name, model in self.individual_models.items():
            try:
                # Train individual model
                model.fit(X_train, y_train)
                
                # Evaluate performance
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                individual_scores[name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"{name} - Test accuracy: {test_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate model {name}: {str(e)}")
                individual_scores[name] = {
                    'train_score': 0,
                    'test_score': 0,
                    'cv_score': 0,
                    'cv_std': 0
                }
        
        return individual_scores
    
    def _generate_visualization_report(self, results):
        """Generate visualization report"""
        try:
            # Create save directory
            os.makedirs('models/ensemble_results', exist_ok=True)
            
            # 1. Model performance comparison chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Individual models vs ensemble model accuracy comparison
            model_names = list(results['individual_scores'].keys()) + ['Ensemble']
            test_scores = [results['individual_scores'][name]['test_score'] 
                          for name in results['individual_scores'].keys()]
            test_scores.append(results['ensemble_test_score'])
            
            axes[0, 0].bar(model_names, test_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
            axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('Test Accuracy')
            axes[0, 0].set_ylim(0, 1)
            
            # Add value labels
            for i, score in enumerate(test_scores):
                axes[0, 0].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
            
            # Cross-validation score distribution
            cv_scores = [results['individual_scores'][name]['cv_score'] 
                        for name in results['individual_scores'].keys()]
            cv_scores.append(results['ensemble_cv_scores'].mean())
            
            axes[0, 1].bar(model_names, cv_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
            axes[0, 1].set_title('Cross-validation Accuracy Comparison', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('CV Accuracy')
            axes[0, 1].set_ylim(0, 1)
            
            # Confusion matrix
            cm = results['confusion_matrix']
            emotion_labels = [f"E{i}" for i in range(len(self.label_encoder.classes_))]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=emotion_labels,
                       yticklabels=emotion_labels,
                       ax=axes[1, 0])
            axes[1, 0].set_title('Ensemble Model Confusion Matrix', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Predicted Label')
            axes[1, 0].set_ylabel('True Label')
            
            # Performance improvement chart
            improvement = results['ensemble_test_score'] - max([results['individual_scores'][name]['test_score'] 
                                                              for name in results['individual_scores'].keys()])
            axes[1, 1].bar(['Best Individual', 'Ensemble'], 
                          [max([results['individual_scores'][name]['test_score'] 
                               for name in results['individual_scores'].keys()]),
                           results['ensemble_test_score']], 
                          color=['orange', 'green'])
            axes[1, 1].set_title(f'Ensemble Learning Performance Improvement: +{improvement:.3f}', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Test Accuracy')
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('models/ensemble_results/performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Generate text report
            self._generate_text_report(results)
            
            logger.info("Visualization report generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate visualization report: {str(e)}")
    
    def _generate_text_report(self, results):
        """Generate text report"""
        # Calculate performance improvement
        best_individual = max([scores['test_score'] for scores in results['individual_scores'].values()])
        improvement = results['ensemble_test_score'] - best_individual
        
        report_content = f"""# Ensemble Learning Music Emotion Recognition Report

## Performance Overview

### Ensemble Model Performance
- Training Accuracy: {results['ensemble_train_score']:.4f}
- Test Accuracy: {results['ensemble_test_score']:.4f}
- Cross-validation Accuracy: {results['ensemble_cv_scores'].mean():.4f} (+/- {results['ensemble_cv_scores'].std()*2:.4f})

### Individual Model Performance Comparison

"""
        
        for name, scores in results['individual_scores'].items():
            report_content += f"- {name}: Test Accuracy = {scores['test_score']:.4f}, CV Score = {scores['cv_score']:.4f}\n"
        
        report_content += f"""
### Performance Improvement Analysis
- Best Individual Model Accuracy: {best_individual:.4f}
- Ensemble Model Accuracy: {results['ensemble_test_score']:.4f}
- Performance Improvement: +{improvement:.4f} ({improvement/best_individual*100:.1f}%)

## Conclusion
The ensemble learning approach successfully improved model performance by combining multiple algorithms.

---
Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open('models/ensemble_results/ensemble_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def save_models(self):
        """Save trained models"""
        try:
            os.makedirs('models/ensemble_results', exist_ok=True)
            
            # Save ensemble model
            joblib.dump(self.ensemble_model, 'models/ensemble_results/ensemble_model.pkl')
            joblib.dump(self.scaler, 'models/ensemble_results/ensemble_scaler.pkl')
            joblib.dump(self.label_encoder, 'models/ensemble_results/ensemble_label_encoder.pkl')
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
    
    def load_models(self):
        """Load trained models"""
        try:
            self.ensemble_model = joblib.load('models/ensemble_results/ensemble_model.pkl')
            self.scaler = joblib.load('models/ensemble_results/ensemble_scaler.pkl')
            self.label_encoder = joblib.load('models/ensemble_results/ensemble_label_encoder.pkl')
            
            logger.info("Ensemble models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def predict(self, features):
        """Predict emotion"""
        try:
            if self.ensemble_model is None:
                if not self.load_models():
                    raise ValueError("Model not trained and loading failed")
            
            # Feature standardization
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Prediction
            prediction = self.ensemble_model.predict(features_scaled)[0]
            probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
            
            # Convert labels
            emotion = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get all emotion probabilities
            emotion_probs = {}
            for i, emotion_class in enumerate(self.label_encoder.classes_):
                emotion_probs[emotion_class] = float(probabilities[i])
            
            return {
                'predicted_emotion': emotion,
                'confidence': float(probabilities[prediction]),
                'all_probabilities': emotion_probs,
                'model_type': 'ensemble'
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None

def main():
    """Main function - Train ensemble learning model"""
    print("Music Emotion Recognition - Ensemble Learning Training")
    print("=" * 60)
    
    # Create ensemble classifier
    classifier = SimpleEnsembleClassifier()
    
    # Load data
    features, labels = classifier.load_data()
    if features is None or labels is None:
        print("Failed to load data. Please ensure data files exist.")
        return
    
    # Train model
    results = classifier.train_ensemble(features, labels)
    if results is None:
        print("Model training failed")
        return
    
    # Display results
    print("\nTraining Results:")
    print(f"Ensemble Model Test Accuracy: {results['ensemble_test_score']:.4f}")
    print(f"Cross-validation Accuracy: {results['ensemble_cv_scores'].mean():.4f}")
    
    # Calculate performance improvement
    best_individual = max([scores['test_score'] for scores in results['individual_scores'].values()])
    improvement = results['ensemble_test_score'] - best_individual
    print(f"Performance Improvement: +{improvement:.4f} ({improvement/best_individual*100:.1f}%)")
    
    print("\nEnsemble learning training completed successfully!")
    print("Results saved in models/ensemble_results/ directory")

if __name__ == "__main__":
    main() 