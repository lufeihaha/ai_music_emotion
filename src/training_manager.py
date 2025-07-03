import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
import torch

from .audio_processor import AudioProcessor
from .emotion_classifier import EmotionClassifier, TraditionalClassifier
from .data_augmentation import DataAugmentor
from .visualization import Visualizer
from .model_tuning import ModelTuner

class TrainingManager:
    """训练管理器：整合所有功能的高级接口"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化训练管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_components()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        default_config = {
            'sample_rate': 22050,
            'duration': None,
            'data_augmentation': {
                'enabled': True,
                'methods': ['noise', 'pitch', 'stretch'],
                'augmentations_per_class': {'happy': 10, 'sad': 10, 'angry': 10, 'peaceful': 10}
            },
            'model': {
                'type': 'deep',  # 'deep' 或 'traditional'
                'params': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42
            },
            'tuning': {
                'enabled': True,
                'n_trials': 100,
                'cv': 5
            },
            'output_dir': 'outputs'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_components(self):
        """初始化所有组件"""
        self.audio_processor = AudioProcessor(
            sr=self.config['sample_rate'],
            duration=self.config['duration']
        )
        
        self.augmentor = DataAugmentor(sr=self.config['sample_rate'])
        self.visualizer = Visualizer()
        self.model_tuner = ModelTuner(log_dir=os.path.join(self.config['output_dir'], 'tuning_logs'))
        
        if self.config['model']['type'] == 'deep':
            self.model = EmotionClassifier()
        else:
            self.model = TraditionalClassifier()
    
    def prepare_data(self, audio_paths: List[str], labels: List[str]) -> Dict[str, np.ndarray]:
        """
        准备训练数据
        
        Args:
            audio_paths: 音频文件路径列表
            labels: 标签列表
            
        Returns:
            包含训练、验证和测试集的字典
        """
        self.logger.info("开始准备数据...")
        
        # 提取特征
        X, y = self.audio_processor.preprocess_for_training(audio_paths, labels)
        
        # 数据增强
        if self.config['data_augmentation']['enabled']:
            self.logger.info("正在进行数据增强...")
            X, y = self.augmentor.augment_dataset(
                X, y,
                self.config['data_augmentation']['augmentations_per_class']
            )
        
        # 划分数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['training']['validation_size'],
            random_state=self.config['training']['random_state']
        )
        
        # 可视化数据分布
        self.visualizer.plot_emotion_distribution(y_train, list(set(labels)))
        self.visualizer.save_figure(os.path.join(self.config['output_dir'], 'emotion_distribution.png'))
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def train_model(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data: 数据集字典
            
        Returns:
            训练结果
        """
        self.logger.info("开始训练模型...")
        
        # 模型调优
        if self.config['tuning']['enabled']:
            self.logger.info("正在进行超参数调优...")
            if self.config['model']['type'] == 'deep':
                param_ranges = {
                    'learning_rate': (0.0001, 0.01),
                    'batch_size': (16, 128),
                    'epochs': (50, 200)
                }
                tuning_results = self.model_tuner.tune_deep_model(
                    self.model.__class__,
                    param_ranges,
                    data['X_train'],
                    data['y_train'],
                    n_trials=self.config['tuning']['n_trials']
                )
            else:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
                tuning_results = self.model_tuner.tune_traditional_model(
                    self.model.__class__,
                    param_grid,
                    data['X_train'],
                    data['y_train'],
                    cv=self.config['tuning']['cv']
                )
            
            # 更新模型参数
            self.config['model']['params'].update(tuning_results['best_params'])
        
        # 训练模型
        history = self.model.train_model(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val'],
            **self.config['model']['params']
        )
        
        # 可视化训练过程
        if history:
            self.visualizer.plot_training_history(
                history,
                ['loss', 'val_loss', 'accuracy', 'val_accuracy']
            )
            self.visualizer.save_figure(os.path.join(self.config['output_dir'], 'training_history.png'))
        
        return {'history': history, 'tuning_results': tuning_results if self.config['tuning']['enabled'] else None}
    
    def evaluate_model(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            data: 数据集字典
            
        Returns:
            评估结果
        """
        self.logger.info("开始评估模型...")
        
        # 在测试集上评估
        test_accuracy = self.model.evaluate(data['X_test'], data['y_test'])
        
        # 获取预测结果
        predictions = self.model.predict(data['X_test'])
        probabilities = self.model.predict_proba(data['X_test'])
        
        # 可视化结果
        self.visualizer.plot_confusion_matrix(
            data['y_test'],
            predictions,
            list(set(data['y_test']))
        )
        self.visualizer.save_figure(os.path.join(self.config['output_dir'], 'confusion_matrix.png'))
        
        self.visualizer.plot_roc_curves(
            data['y_test'],
            probabilities,
            list(set(data['y_test']))
        )
        self.visualizer.save_figure(os.path.join(self.config['output_dir'], 'roc_curves.png'))
        
        return {
            'test_accuracy': test_accuracy,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def analyze_features(self, data: Dict[str, np.ndarray], feature_names: List[str]):
        """
        分析特征
        
        Args:
            data: 数据集字典
            feature_names: 特征名称列表
        """
        self.logger.info("开始分析特征...")
        
        # 特征相关性分析
        self.visualizer.plot_feature_correlations(
            data['X_train'],
            feature_names
        )
        self.visualizer.save_figure(os.path.join(self.config['output_dir'], 'feature_correlations.png'))
        
        # 特征重要性分析（仅适用于传统模型）
        if hasattr(self.model, 'feature_importances_'):
            self.visualizer.plot_feature_importance(
                feature_names,
                self.model.feature_importances_
            )
            self.visualizer.save_figure(os.path.join(self.config['output_dir'], 'feature_importance.png'))
    
    def save_results(self, results: Dict[str, Any]):
        """
        保存所有结果
        
        Args:
            results: 结果字典
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(self.config['output_dir'], f'results_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(results_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # 保存结果摘要
        summary = {
            'test_accuracy': float(results['evaluation']['test_accuracy']),
            'model_type': self.config['model']['type'],
            'data_augmentation': self.config['data_augmentation']['enabled'],
            'tuning_enabled': self.config['tuning']['enabled'],
            'timestamp': timestamp
        }
        
        with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # 保存模型
        if isinstance(self.model, EmotionClassifier):
            self.model.save_model(
                os.path.join(results_dir, 'model.pt'),
                os.path.join(results_dir, 'encoder.pkl')
            )
        else:
            self.model.save_model(os.path.join(results_dir, 'model.pkl'))
        
        self.logger.info(f"所有结果已保存到: {results_dir}")
    
    def run_pipeline(self, audio_paths: List[str], labels: List[str],
                    feature_names: List[str]) -> Dict[str, Any]:
        """
        运行完整的训练和评估流程
        
        Args:
            audio_paths: 音频文件路径列表
            labels: 标签列表
            feature_names: 特征名称列表
            
        Returns:
            所有结果的字典
        """
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 准备数据
        data = self.prepare_data(audio_paths, labels)
        
        # 训练模型
        training_results = self.train_model(data)
        
        # 评估模型
        evaluation_results = self.evaluate_model(data)
        
        # 分析特征
        self.analyze_features(data, feature_names)
        
        # 整合所有结果
        results = {
            'training': training_results,
            'evaluation': evaluation_results
        }
        
        # 保存结果
        self.save_results(results)
        
        return results 