#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicPredictor:
    """动态预测器 - 自动选择最佳可用模型"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.model_type = None
        self.model_info = {}
        
        # 按优先级排序的模型路径
        self.model_paths = [
            {
                'name': 'improved_ensemble',
                'path': 'models/improved_ensemble',
                'files': ['ensemble_model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'feature_selector.pkl'],
                'description': '改进的集成学习模型'
            },
            {
                'name': 'chinese_optimized',
                'path': 'models/chinese_optimized',
                'files': ['randomforest_model.pkl', 'scaler.pkl', 'label_encoder.pkl'],
                'description': '中文优化随机森林模型'
            },
            {
                'name': 'unified_results',
                'path': 'models/unified_results',
                'files': ['randomforest_model.pkl', 'scaler.pkl', 'label_encoder.pkl'],
                'description': '统一训练随机森林模型'
            },
            {
                'name': 'base_models',
                'path': 'models',
                'files': ['randomforest_model.pkl', 'scaler.pkl', 'label_encoder.pkl'],
                'description': '基础随机森林模型'
            }
        ]
        
        self._load_best_model()
    
    def _check_model_availability(self, model_config):
        """检查模型是否可用"""
        try:
            for file in model_config['files']:
                file_path = os.path.join(model_config['path'], file)
                if not os.path.exists(file_path):
                    return False
            return True
        except Exception:
            return False
    
    def _load_best_model(self):
        """加载最佳可用模型"""
        for model_config in self.model_paths:
            if self._check_model_availability(model_config):
                try:
                    self._load_model(model_config)
                    logger.info(f"✅ 成功加载模型: {model_config['description']}")
                    return
                except Exception as e:
                    logger.warning(f"⚠️ 加载模型失败 {model_config['name']}: {e}")
                    continue
        
        # 如果所有模型都失败，尝试加载任何可用的模型文件
        self._load_fallback_model()
    
    def _load_model(self, model_config):
        """加载指定的模型配置"""
        model_path = model_config['path']
        
        # 加载模型文件
        if 'ensemble_model.pkl' in model_config['files']:
            self.model = joblib.load(os.path.join(model_path, 'ensemble_model.pkl'))
            self.model_type = 'ensemble'
        else:
            self.model = joblib.load(os.path.join(model_path, 'randomforest_model.pkl'))
            self.model_type = 'random_forest'
        
        # 加载预处理器
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
        
        # 加载特征选择器（如果存在）
        feature_selector_path = os.path.join(model_path, 'feature_selector.pkl')
        if os.path.exists(feature_selector_path):
            self.feature_selector = joblib.load(feature_selector_path)
        
        # 保存模型信息
        self.model_info = {
            'name': model_config['name'],
            'description': model_config['description'],
            'type': self.model_type,
            'path': model_path,
            'has_feature_selector': self.feature_selector is not None
        }
    
    def _load_fallback_model(self):
        """加载备用模型"""
        try:
            # 尝试加载任何可用的模型文件
            model_files = [
                'models/randomforest_model.pkl',
                'models/enhanced_randomforest_model.pkl',
                'models/svm_model.pkl',
                'models/gradientboosting_model.pkl'
            ]
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    self.model = joblib.load(model_file)
                    self.model_type = 'fallback'
                    
                    # 尝试加载对应的预处理器
                    if os.path.exists('models/scaler.pkl'):
                        self.scaler = joblib.load('models/scaler.pkl')
                    if os.path.exists('models/label_encoder.pkl'):
                        self.label_encoder = joblib.load('models/label_encoder.pkl')
                    
                    self.model_info = {
                        'name': 'fallback',
                        'description': f'备用模型: {os.path.basename(model_file)}',
                        'type': self.model_type,
                        'path': model_file
                    }
                    
                    logger.info(f"✅ 加载备用模型: {model_file}")
                    return
            
            raise Exception("没有找到任何可用的模型文件")
            
        except Exception as e:
            logger.error(f"❌ 无法加载任何模型: {e}")
            raise
    
    def predict(self, features):
        """预测情感"""
        if self.model is None:
            raise Exception("模型未加载")
        
        try:
            # 预处理特征
            if len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)
            elif len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # 特征选择
            if self.feature_selector is not None:
                features = self.feature_selector.transform(features)
            
            # 标准化
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # 预测
            predictions = self.model.predict(features)
            
            # 获取概率（如果支持）
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)
            
            # 解码标签
            if self.label_encoder is not None:
                emotion_labels = self.label_encoder.inverse_transform(predictions)
            else:
                emotion_labels = predictions
            
            return {
                'predictions': emotion_labels,
                'probabilities': probabilities,
                'model_info': self.model_info
            }
            
        except Exception as e:
            logger.error(f"❌ 预测失败: {e}")
            raise
    
    def get_model_info(self):
        """获取当前模型信息"""
        return self.model_info
    
    def get_emotion_classes(self):
        """获取情感类别"""
        if self.label_encoder is not None:
            return self.label_encoder.classes_
        else:
            return ['calm', 'energetic', 'happy', 'melancholic']  # 默认类别

# 创建全局预测器实例
_predictor = None

def get_predictor():
    """获取全局预测器实例"""
    global _predictor
    if _predictor is None:
        _predictor = DynamicPredictor()
    return _predictor

def predict_emotion(features):
    """便捷的情感预测函数"""
    predictor = get_predictor()
    return predictor.predict(features)

def get_model_status():
    """获取当前模型状态"""
    try:
        predictor = get_predictor()
        return {
            'status': 'loaded',
            'model_info': predictor.get_model_info(),
            'emotion_classes': list(predictor.get_emotion_classes())
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

if __name__ == "__main__":
    # 测试动态预测器
    try:
        predictor = DynamicPredictor()
        status = get_model_status()
        
        print("🔍 动态预测器状态:")
        print(f"  状态: {status['status']}")
        if status['status'] == 'loaded':
            model_info = status['model_info']
            print(f"  模型: {model_info['description']}")
            print(f"  类型: {model_info['type']}")
            print(f"  情感类别: {status['emotion_classes']}")
            print(f"  特征选择: {model_info.get('has_feature_selector', False)}")
        else:
            print(f"  错误: {status['error']}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}") 