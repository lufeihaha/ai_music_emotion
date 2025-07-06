#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

class AdvancedEmotionClassifier:
    def __init__(self):
        self.detailed_emotions = {
            # 原始4类基础情感
            'calm': {
                'name': '平静',
                'subcategories': ['peaceful', 'relaxed', 'serene'],
                'icon': 'fas fa-leaf',
                'color': '#28a745'
            },
            'happy': {
                'name': '快乐',
                'subcategories': ['joyful', 'cheerful', 'excited'],
                'icon': 'fas fa-smile',
                'color': '#ffc107'
            },
            'energetic': {
                'name': '充满活力',
                'subcategories': ['dynamic', 'powerful', 'intense'],
                'icon': 'fas fa-bolt',
                'color': '#dc3545'
            },
            'melancholic': {
                'name': '忧郁',
                'subcategories': ['sad', 'nostalgic', 'wistful'],
                'icon': 'fas fa-cloud-rain',
                'color': '#6f42c1'
            },
            
            # 新增细化情感类别
            'romantic': {
                'name': '浪漫',
                'subcategories': ['loving', 'tender', 'intimate'],
                'icon': 'fas fa-heart',
                'color': '#e91e63'
            },
            'nostalgic': {
                'name': '怀念',
                'subcategories': ['reminiscent', 'wistful', 'longing'],
                'icon': 'fas fa-clock',
                'color': '#795548'
            },
            'dramatic': {
                'name': '戏剧性',
                'subcategories': ['epic', 'cinematic', 'theatrical'],
                'icon': 'fas fa-theater-masks',
                'color': '#9c27b0'
            },
            'mysterious': {
                'name': '神秘',
                'subcategories': ['enigmatic', 'dark', 'suspenseful'],
                'icon': 'fas fa-mask',
                'color': '#424242'
            }
        }
        
        self.emotion_mapping = {
            # 音乐特征到情感的映射规则
            'tempo_slow_harmony_minor': ['melancholic', 'nostalgic', 'romantic'],
            'tempo_slow_harmony_major': ['calm', 'romantic'],
            'tempo_fast_harmony_minor': ['dramatic', 'mysterious'],
            'tempo_fast_harmony_major': ['happy', 'energetic'],
            'high_energy_complex': ['energetic', 'dramatic'],
            'low_energy_simple': ['calm', 'melancholic']
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def create_compound_labels(self, features_df):
        """
        基于音乐特征创建复合情感标签
        """
        compound_labels = []
        
        for idx, row in features_df.iterrows():
            # 分析音乐特征
            tempo = row.get('tempo', 120)
            energy = row.get('energy', 0.5)
            valence = row.get('valence', 0.5)  # 音乐的积极性
            mode = row.get('mode', 1)  # 大调(1)或小调(0)
            
            # 基于特征组合确定情感
            emotions = []
            
            # 节拍分析
            if tempo < 90:
                if mode == 0:  # 小调
                    emotions.extend(['melancholic', 'nostalgic'])
                else:  # 大调
                    emotions.extend(['calm', 'romantic'])
            elif tempo > 140:
                if energy > 0.7:
                    emotions.extend(['energetic', 'dramatic'])
                else:
                    emotions.extend(['happy'])
            else:  # 中等节拍
                if valence > 0.6:
                    emotions.extend(['happy'])
                elif valence < 0.4:
                    emotions.extend(['melancholic'])
                else:
                    emotions.extend(['calm'])
            
            # 选择最合适的情感
            if emotions:
                primary_emotion = emotions[0]
                compound_labels.append(primary_emotion)
            else:
                compound_labels.append('calm')  # 默认
        
        return compound_labels
    
    def train_hierarchical_model(self, features, labels):
        """
        训练层次化情感分类模型
        """
        # 首先训练粗粒度分类器（4个基本情感）
        basic_emotions = ['calm', 'happy', 'energetic', 'melancholic']
        basic_labels = []
        
        for label in labels:
            if label in basic_emotions:
                basic_labels.append(label)
            else:
                # 将细化情感映射到基本情感
                for basic_emotion, config in self.detailed_emotions.items():
                    if basic_emotion in basic_emotions and label in config.get('subcategories', []):
                        basic_labels.append(basic_emotion)
                        break
                else:
                    basic_labels.append('calm')  # 默认映射
        
        # 训练基础分类器
        X_train, X_test, y_train, y_test = train_test_split(
            features, basic_labels, test_size=0.2, random_state=42, stratify=basic_labels
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 编码标签
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # 评估模型
        y_pred = self.model.predict(X_test_scaled)
        
        logging.info("基础情感分类报告:")
        logging.info(classification_report(y_test_encoded, y_pred, 
                                         target_names=self.label_encoder.classes_))
        
        return self.model
    
    def predict_detailed_emotion(self, features):
        """
        预测详细的情感类别，包括置信度和情感强度
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 标准化特征
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 基础预测
        basic_prediction = self.model.predict(features_scaled)[0]
        basic_probabilities = self.model.predict_proba(features_scaled)[0]
        basic_emotion = self.label_encoder.inverse_transform([basic_prediction])[0]
        
        # 分析音乐特征以细化情感
        detailed_analysis = self._analyze_detailed_features(features)
        
        # 构建完整的情感分析结果
        result = {
            'primary_emotion': basic_emotion,
            'confidence': float(basic_probabilities[basic_prediction]),
            'emotion_details': self.detailed_emotions.get(basic_emotion, {}),
            'feature_analysis': detailed_analysis,
            'all_probabilities': {
                emotion: float(prob) 
                for emotion, prob in zip(self.label_encoder.classes_, basic_probabilities)
            },
            'intensity': self._calculate_intensity(features),
            'mood_descriptors': self._generate_mood_descriptors(basic_emotion, detailed_analysis)
        }
        
        return result
    
    def _analyze_detailed_features(self, features):
        """
        分析详细的音乐特征
        """
        # 这里简化处理，实际应该根据特征向量的具体含义来分析
        energy_level = np.mean(features[:13]) if len(features) > 13 else 0.5  # MFCC平均值
        spectral_info = np.mean(features[13:20]) if len(features) > 20 else 0.5  # 频谱特征
        
        return {
            'energy_level': float(energy_level),
            'spectral_complexity': float(spectral_info),
            'dynamic_range': float(np.std(features[:20]) if len(features) > 20 else 0.3)
        }
    
    def _calculate_intensity(self, features):
        """
        计算情感强度（0-1）
        """
        # 基于特征的方差和均值计算强度
        intensity = min(1.0, np.std(features) * 2)
        return float(intensity)
    
    def _generate_mood_descriptors(self, emotion, analysis):
        """
        生成情感描述词
        """
        descriptors = []
        
        config = self.detailed_emotions.get(emotion, {})
        base_descriptors = config.get('subcategories', [emotion])
        
        # 根据分析结果选择描述词
        if analysis['energy_level'] > 0.7:
            descriptors.extend(['强烈的', '充满活力的'])
        elif analysis['energy_level'] < 0.3:
            descriptors.extend(['轻柔的', '温和的'])
        
        if analysis['spectral_complexity'] > 0.6:
            descriptors.extend(['复杂的', '丰富的'])
        else:
            descriptors.extend(['简洁的', '纯净的'])
        
        return descriptors + base_descriptors
    
    def save_model(self, model_dir="models/advanced"):
        """
        保存高级模型
        """
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_dir, 'advanced_emotion_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'advanced_scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'advanced_label_encoder.pkl'))
        
        # 保存情感配置
        import json
        with open(os.path.join(model_dir, 'emotion_config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.detailed_emotions, f, ensure_ascii=False, indent=2)
        
        logging.info(f"高级模型已保存到 {model_dir}")
    
    def load_model(self, model_dir="models/advanced"):
        """
        加载高级模型
        """
        self.model = joblib.load(os.path.join(model_dir, 'advanced_emotion_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'advanced_scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'advanced_label_encoder.pkl'))
        
        # 加载情感配置
        import json
        config_path = os.path.join(model_dir, 'emotion_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.detailed_emotions = json.load(f)
        
        logging.info(f"高级模型已从 {model_dir} 加载")

if __name__ == "__main__":
    classifier = AdvancedEmotionClassifier()
    print("🎭 高级情感分类器已初始化")
    print(f"📊 支持的情感类别: {list(classifier.detailed_emotions.keys())}")
    
    # 这里可以添加训练和测试代码
    print("💡 使用方法:")
    print("1. 准备特征数据和标签")
    print("2. 调用 train_hierarchical_model() 训练模型")
    print("3. 使用 predict_detailed_emotion() 进行预测")
    print("4. 调用 save_model() 保存模型") 