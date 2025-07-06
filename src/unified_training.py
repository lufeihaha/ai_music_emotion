#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import logging
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UnifiedMusicEmotionTrainer:
    def __init__(self):
        self.setup_environment()
        
    def setup_environment(self):
        os.makedirs('models/unified_results', exist_ok=True)
        os.makedirs('models/unified_results/visualizations', exist_ok=True)
        
    def load_and_prepare_data(self):
        logging.info("Loading and preprocessing data...")
        
        try:
            features = np.load('data/features.npy')
            labels_df = pd.read_csv('data/processed/emotion_labels.csv')
            labels = labels_df['emotion'].values
            
            logging.info(f"Original feature shape: {features.shape}")
            
            if len(features.shape) == 3:
                n_samples, n_segments, n_features = features.shape
                features_2d = features.reshape(n_samples, n_segments * n_features)
                logging.info(f"Reshaped features to 2D: {features_2d.shape}")
            else:
                features_2d = features
                
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)
            
            X_train, X_test, y_train, y_test = train_test_split(
                features_2d, labels_encoded, 
                test_size=0.2, random_state=42, 
                stratify=labels_encoded
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            joblib.dump(scaler, 'models/unified_results/scaler.pkl')
            joblib.dump(label_encoder, 'models/unified_results/label_encoder.pkl')
            
            self.label_encoder = label_encoder
            
            return {
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test
            }
            
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            return None
    
    def create_traditional_models(self):
        return {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, n_jobs=-1, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), activation='relu',
                solver='adam', max_iter=1000, random_state=42, early_stopping=True
            )
        }
    
    def train_traditional_models(self, data):
        logging.info("Training traditional ML models...")
        
        models = self.create_traditional_models()
        results = {}
        
        for name, model in models.items():
            logging.info(f"Training {name} model...")
            
            model.fit(data['X_train_scaled'], data['y_train'])
            cv_scores = cross_val_score(model, data['X_train_scaled'], data['y_train'], cv=5)
            y_pred = model.predict(data['X_test_scaled'])
            test_accuracy = accuracy_score(data['y_test'], y_pred)
            test_f1 = f1_score(data['y_test'], y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'predictions': y_pred
            }
            
            joblib.dump(model, f'models/unified_results/{name.lower()}_model.pkl')
            logging.info(f"{name} - CV: {cv_scores.mean():.4f}(±{cv_scores.std():.4f}), Test: {test_accuracy:.4f}")
        
        return results
    
    def train_deep_model(self, data):
        logging.info("Training deep learning model...")
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, callbacks
            
            input_shape = (data['X_train_scaled'].shape[1],)
            num_classes = len(self.label_encoder.classes_)
            
            model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model_callbacks = [
                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            history = model.fit(
                data['X_train_scaled'], data['y_train'],
                validation_data=(data['X_test_scaled'], data['y_test']),
                epochs=50,
                batch_size=32,
                callbacks=model_callbacks,
                verbose=1
            )
            
            test_loss, test_accuracy = model.evaluate(data['X_test_scaled'], data['y_test'], verbose=0)
            y_pred = np.argmax(model.predict(data['X_test_scaled']), axis=1)
            test_f1 = f1_score(data['y_test'], y_pred, average='weighted')
            
            model.save('models/unified_results/deep_model.keras')
            
            result = {
                'model': model,
                'history': history.history,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'predictions': y_pred
            }
            
            logging.info(f"Deep Learning Model - Test Accuracy: {test_accuracy:.4f}")
            return result
            
        except ImportError:
            logging.warning("TensorFlow not available, skipping deep learning model")
            return None
    
    def visualize_results(self, traditional_results, deep_results, data):
        logging.info("Generating visualizations...")
        
        model_names = list(traditional_results.keys())
        accuracies = [traditional_results[name]['test_accuracy'] for name in model_names]
        f1_scores = [traditional_results[name]['test_f1'] for name in model_names]
        
        if deep_results:
            model_names.append('Deep Learning')
            accuracies.append(deep_results['test_accuracy'])
            f1_scores.append(deep_results['test_f1'])
        
        # Model comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple']
        bars1 = ax1.bar(model_names, accuracies, color=colors[:len(model_names)])
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        bars2 = ax2.bar(model_names, f1_scores, color=colors[:len(model_names)])
        ax2.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('models/unified_results/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion matrices
        emotion_labels = ['calm', 'energetic', 'happy', 'melancholic']
        n_models = len(traditional_results) + (1 if deep_results else 0)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (name, results) in enumerate(traditional_results.items()):
            cm = confusion_matrix(data['y_test'], results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=emotion_labels,
                       yticklabels=emotion_labels,
                       ax=axes[i])
            axes[i].set_title(f'{name}\nAccuracy: {results["test_accuracy"]:.3f}')
        
        if deep_results:
            cm = confusion_matrix(data['y_test'], deep_results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=emotion_labels,
                       yticklabels=emotion_labels,
                       ax=axes[len(traditional_results)])
            axes[len(traditional_results)].set_title(f'Deep Learning\nAccuracy: {deep_results["test_accuracy"]:.3f}')
        
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('models/unified_results/visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, traditional_results, deep_results):
        logging.info("Generating comprehensive performance report...")
        
        best_score = 0
        best_model_name = None
        
        print("\n" + "="*60)
        print("Music Emotion Recognition Model Performance Summary")
        print("="*60)
        print("\nTraditional Machine Learning Models:")
        
        for name, results in traditional_results.items():
            print(f"{name:>15}: Accuracy={results['test_accuracy']:.4f}, F1={results['test_f1']:.4f}")
            if results['test_accuracy'] > best_score:
                best_score = results['test_accuracy']
                best_model_name = name
        
        if deep_results:
            print(f"\nDeep Learning Model:")
            print(f"{'Deep Learning':>15}: Accuracy={deep_results['test_accuracy']:.4f}, F1={deep_results['test_f1']:.4f}")
            
            if deep_results['test_accuracy'] > best_score:
                best_score = deep_results['test_accuracy']
                best_model_name = 'Deep Learning'
        
        print(f"\nBest Model: {best_model_name} (Accuracy: {best_score:.4f})")
        print("="*60)
        
        return best_model_name, best_score
    
    def run_complete_training(self):
        logging.info("Starting complete model training and comparison pipeline...")
        
        data = self.load_and_prepare_data()
        if data is None:
            return None
        
        traditional_results = self.train_traditional_models(data)
        deep_results = self.train_deep_model(data)
        self.visualize_results(traditional_results, deep_results, data)
        best_model, best_score = self.generate_report(traditional_results, deep_results)
        
        logging.info("Training pipeline completed! Results saved to models/unified_results/")
        
        return {
            'traditional_results': traditional_results,
            'deep_results': deep_results,
            'best_model': best_model,
            'best_score': best_score
        }

if __name__ == "__main__":
    trainer = UnifiedMusicEmotionTrainer()
    results = trainer.run_complete_training() 