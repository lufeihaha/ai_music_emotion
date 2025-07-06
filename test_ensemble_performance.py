#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

def test_ensemble_performance():
    """测试集成模型性能"""
    print("🚀 测试改进的集成学习模型性能")
    print("=" * 50)
    
    try:
        # 加载数据
        features = np.load('data/features.npy')
        labels_df = pd.read_csv('data/processed/emotion_labels.csv')
        labels = labels_df['emotion'].values
        
        print(f"📊 数据加载完成 - 特征: {features.shape}, 标签: {len(labels)}")
        
        # 重塑特征
        if len(features.shape) == 3:
            features = features.reshape(features.shape[0], -1)
        
        # 加载集成模型和预处理器
        model_path = 'models/improved_ensemble'
        ensemble_model = joblib.load(f'{model_path}/ensemble_model.pkl')
        scaler = joblib.load(f'{model_path}/scaler.pkl')
        label_encoder = joblib.load(f'{model_path}/label_encoder.pkl')
        feature_selector = joblib.load(f'{model_path}/feature_selector.pkl')
        
        print(f"✅ 集成模型加载成功")
        print(f"   - 模型类型: {type(ensemble_model).__name__}")
        print(f"   - 特征选择器: {type(feature_selector).__name__}")
        
        # 标签编码
        y_encoded = label_encoder.transform(labels)
        
        # 数据分割（使用相同的随机种子确保一致性）
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # 特征选择
        X_train_selected = feature_selector.transform(X_train)
        X_test_selected = feature_selector.transform(X_test)
        
        # 特征标准化
        X_train_scaled = scaler.transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        print(f"📈 数据预处理完成")
        print(f"   - 训练集: {X_train_scaled.shape}")
        print(f"   - 测试集: {X_test_scaled.shape}")
        
        # 预测
        y_pred = ensemble_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 集成模型性能:")
        print(f"   - 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 详细分类报告
        emotion_names = label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=emotion_names, output_dict=True)
        
        print(f"\n📋 详细分类报告:")
        for emotion in emotion_names:
            metrics = report[emotion]
            print(f"   {emotion}: 精确率={metrics['precision']:.3f}, "
                  f"召回率={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # 与之前的最佳模型比较
        print(f"\n📊 性能对比:")
        print(f"   - 之前最佳 (SVM): 67.67%")
        print(f"   - 集成模型: {accuracy*100:.2f}%")
        
        improvement = accuracy - 0.6767
        if improvement > 0:
            print(f"   - 🎉 性能提升: +{improvement*100:.2f}%")
        else:
            print(f"   - ⚠️ 性能变化: {improvement*100:+.2f}%")
        
        return accuracy
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_ensemble_performance() 