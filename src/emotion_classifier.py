import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, cast
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionNet(nn.Module):
    def __init__(self, input_size, num_emotions):
        super(EmotionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_emotions),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.model(x)

class EmotionClassifier:
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'peaceful', 'excited']
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self, input_shape):
        """构建PyTorch模型"""
        self.model = EmotionNet(input_shape[0], len(self.emotions))
        self.model.to(self.device)
        return self.model
    
    def prepare_features(self, features):
        """准备特征数据"""
        feature_list = []
        for feature_dict in features:
            feature_vector = np.concatenate([
                feature_dict['mfcc'],
                [feature_dict['spectral_centroid']],
                feature_dict['chroma'],
                [feature_dict['zero_crossing_rate']],
                [feature_dict['rmse']]
            ])
            feature_list.append(feature_vector)
        
        return np.array(feature_list)
    
    def predict_emotion(self, features):
        """预测情感"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # 准备特征
        X = self.prepare_features([features])
        X_scaled = self.scaler.transform(X)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        # 返回情感标签和概率
        emotion_idx = np.argmax(predictions[0])
        return {
            'emotion': self.emotions[emotion_idx],
            'probabilities': {
                emotion: float(prob)
                for emotion, prob in zip(self.emotions, predictions[0])
            }
        }
    
    def predict_emotion_sequence(self, feature_sequence):
        """预测一系列特征的情感变化"""
        emotions = []
        for features in feature_sequence:
            emotion = self.predict_emotion(features)
            emotions.append(emotion)
        return emotions 

class TraditionalClassifier:
    """Traditional machine learning classifier for emotion recognition."""
    
    SUPPORTED_MODELS = {
        'svm': SVC,
        'random_forest': RandomForestClassifier
    }
    
    def __init__(self, model_type: str = 'svm', model_params: Optional[Dict] = None):
        """
        Initialize the classifier.
        
        Args:
            model_type (str): Type of model to use ('svm' or 'random_forest')
            model_params (Dict, optional): Parameters for the model
        """
        self._setup_logging()
        self.model_type = model_type.lower()
        
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type {model_type} not supported. Choose from {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_params = model_params or self._get_default_params()
        self.model: Union[SVC, RandomForestClassifier] = self.SUPPORTED_MODELS[self.model_type](**self.model_params)
        self.scaler = StandardScaler()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_default_params(self) -> Dict:
        """Get default parameters for the selected model type."""
        if self.model_type == 'svm':
            return {
                'kernel': 'rbf',
                'C': 1.0,
                'probability': True
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            }
    
    def _perform_grid_search(self, X: np.ndarray, y: np.ndarray) -> Union[SVC, RandomForestClassifier]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            
        Returns:
            Union[SVC, RandomForestClassifier]: Best model found
        """
        param_grid = self._get_param_grid()
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y)
        
        self.logger.info(f"Best parameters found: {grid_search.best_params_}")
        return cast(Union[SVC, RandomForestClassifier], grid_search.best_estimator_)
    
    def _get_param_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid for grid search based on model type."""
        if self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 1],
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
            }
        return {}  # Empty dict as fallback
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             validation_split: float = 0.2,
             perform_grid_search: bool = False) -> Dict[str, Union[float, Dict[str, Any]]]:
        """
        Train the classifier.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            validation_split (float): Proportion of data to use for validation
            perform_grid_search (bool): Whether to perform grid search for hyperparameter tuning
            
        Returns:
            Dict[str, Union[float, Dict[str, Any]]]: Training results including metrics
        """
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data if validation is requested
            if validation_split > 0:
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
            else:
                X_train, y_train = X_scaled, y
                X_val, y_val = None, None
            
            # Perform grid search if requested
            if perform_grid_search:
                self.model = self._perform_grid_search(X_train, y_train)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Compute metrics
            results: Dict[str, Union[float, Dict[str, Any]]] = {
                'train_score': float(self.model.score(X_train, y_train))
            }
            
            if validation_split > 0 and X_val is not None and y_val is not None:
                results['val_score'] = float(self.model.score(X_val, y_val))
                results['val_report'] = classification_report(
                    y_val, 
                    self.model.predict(X_val),
                    output_dict=True
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict emotions for new data.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Predicted emotions
        """
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for predictions.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Probability estimates for each class
        """
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise
    
    def save_model(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Save the trained model and scaler.
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str, optional): Path to save the scaler
        """
        try:
            joblib.dump(self.model, model_path)
            if scaler_path:
                joblib.dump(self.scaler, scaler_path)
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Load a trained model and scaler.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str, optional): Path to the saved scaler
        """
        try:
            self.model = joblib.load(model_path)
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str], save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str]): Label names
            save_path (str, optional): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 