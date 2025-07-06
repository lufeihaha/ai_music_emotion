#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版本的模型评估脚本
解决了标签类型不匹配问题和中文显示问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import os

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model_and_data():
    """加载模型和测试数据"""
    try:
        # 尝试加载不同类型的模型
        model_paths = [
            'models/improved_ensemble/ensemble_model.pkl',
            'models/randomforest_model.pkl',
            'models/svm_model.pkl',
            'models/gradientboosting_model.pkl'
        ]
        
        model = None
        model_type = None
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    model_type = path.split('/')[-1].replace('_model.pkl', '').replace('.pkl', '')
                    logging.info(f"成功加载模型: {path}")
                    break
                except Exception as e:
                    logging.warning(f"无法加载模型 {path}: {e}")
                    continue
        
        if model is None:
            logging.error("未找到可用的模型文件")
            return None, None, None, None
        
        # 加载测试数据
        features = np.load('data/features.npy')
        labels_df = pd.read_csv('data/processed/emotion_labels.csv')
        
        # 如果特征是3维的，重塑为2维
        if len(features.shape) == 3:
            features = features.reshape(features.shape[0], -1)
            logging.info(f"特征重塑为: {features.shape}")
        
        # 划分测试集（使用相同的随机种子确保一致性）
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            features, labels_df['emotion'].values, 
            test_size=0.2, random_state=42
        )
        
        return model, model_type, X_test, y_test
        
    except Exception as e:
        logging.error(f"加载模型和数据时出错: {e}")
        return None, None, None, None

def preprocess_data_for_model(model, X_test):
    """为模型预处理数据"""
    try:
        # 尝试加载预处理器
        preprocessors = [
            'models/improved_ensemble/scaler.pkl',
            'models/improved_ensemble/feature_selector.pkl',
            'models/scaler.pkl',
            'models/feature_scaler.joblib'
        ]
        
        X_processed = X_test.copy()
        
        # 特征选择
        for selector_path in ['models/improved_ensemble/feature_selector.pkl']:
            if os.path.exists(selector_path):
                try:
                    feature_selector = joblib.load(selector_path)
                    X_processed = feature_selector.transform(X_processed)
                    logging.info(f"应用特征选择: {X_processed.shape}")
                    break
                except:
                    continue
        
        # 标准化
        for scaler_path in preprocessors:
            if 'scaler' in scaler_path and os.path.exists(scaler_path):
                try:
                    scaler = joblib.load(scaler_path)
                    X_processed = scaler.transform(X_processed)
                    logging.info(f"应用数据标准化")
                    break
                except:
                    continue
        
        return X_processed
        
    except Exception as e:
        logging.warning(f"数据预处理失败，使用原始数据: {e}")
        return X_test

def evaluate_model_performance():
    """评估模型性能"""
    try:
        # 加载模型和数据
        model, model_type, X_test, y_test = load_model_and_data()
        if model is None:
            return
        
        # 预处理数据
        X_test_processed = preprocess_data_for_model(model, X_test)
        
        # 定义标签（英文和中文对照）
        labels_en = ['calm', 'energetic', 'happy', 'melancholic']
        labels_cn = ['平静', '活力', '快乐', '忧郁']
        labels_display = [f'{en}\n{cn}' for en, cn in zip(labels_en, labels_cn)]
        
        # 处理标签编码
        label_encoder = LabelEncoder()
        label_encoder.fit(labels_en)
        
        # 统一标签格式
        if isinstance(y_test[0], str):
            y_test_encoded = label_encoder.transform(y_test)
            logging.info("真实标签: 字符串 → 数字")
        else:
            y_test_encoded = y_test.astype(int)
            logging.info("真实标签: 已是数字格式")
        
        # 模型预测
        logging.info(f"使用 {model_type} 模型进行预测...")
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_processed)
            y_pred = model.predict(X_test_processed)
            has_proba = True
        else:
            y_pred = model.predict(X_test_processed)
            has_proba = False
        
        # 确保预测结果是整数
        if isinstance(y_pred[0], str):
            y_pred_encoded = label_encoder.transform(y_pred)
        else:
            y_pred_encoded = y_pred.astype(int)
        
        logging.info(f"预测完成: {len(y_pred_encoded)} 个样本")
        logging.info(f"标签范围 - 真实: [{y_test_encoded.min()}, {y_test_encoded.max()}], 预测: [{y_pred_encoded.min()}, {y_pred_encoded.max()}]")
        
        # 计算基本指标
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
        
        # 打印结果
        print(f"\n{'='*50}")
        print(f"🎯 {model_type.upper()} 模型评估结果")
        print(f"{'='*50}")
        print(f"📊 整体性能:")
        print(f"   准确率 (Accuracy):     {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   加权精确率 (Precision): {precision:.4f}")
        print(f"   加权召回率 (Recall):    {recall:.4f}")
        print(f"   加权F1分数 (F1-Score):  {f1:.4f}")
        
        # 生成详细分类报告
        report = classification_report(
            y_test_encoded, y_pred_encoded, 
            target_names=labels_en, 
            output_dict=True,
            zero_division=0
        )
        
        print(f"\n📋 各情感类别详细指标:")
        print(f"{'情感类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<6}")
        print(f"{'-'*50}")
        
        for i, (label_en, label_cn) in enumerate(zip(labels_en, labels_cn)):
            if label_en in report:
                metrics = report[label_en]
                print(f"{label_en}({label_cn}){'':<3} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} "
                      f"{metrics['f1-score']:<8.4f} {int(metrics['support']):<6}")
        
        # 绘制混淆矩阵（修复中文显示）
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        # 使用中英文双语标签
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels_display, yticklabels=labels_display,
                        cbar_kws={'label': '样本数量'})
        
        plt.title(f'{model_type.upper()} 模型混淆矩阵\nConfusion Matrix', fontsize=16, pad=20)
        plt.xlabel('预测标签 (Predicted Label)', fontsize=12)
        plt.ylabel('真实标签 (True Label)', fontsize=12)
        
        # 调整标签显示
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存结果
        os.makedirs('models/visualizations', exist_ok=True)
        plt.savefig(f'models/visualizations/{model_type}_confusion_matrix_chinese.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 置信度分析（如果有概率预测）
        if has_proba:
            max_probs = np.max(y_pred_proba, axis=1)
            print(f"\n🔍 置信度分析:")
            print(f"   平均置信度: {np.mean(max_probs):.4f}")
            print(f"   置信度标准差: {np.std(max_probs):.4f}")
            print(f"   低置信度样本 (<0.5): {np.sum(max_probs < 0.5)} ({np.sum(max_probs < 0.5)/len(max_probs)*100:.1f}%)")
            print(f"   高置信度样本 (>0.8): {np.sum(max_probs > 0.8)} ({np.sum(max_probs > 0.8)/len(max_probs)*100:.1f}%)")
        
        # 保存评估报告
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'models/{model_type}_evaluation_report_chinese.csv')
        
        print(f"\n✅ 评估完成！")
        print(f"📁 混淆矩阵已保存: models/visualizations/{model_type}_confusion_matrix_chinese.png")
        print(f"📁 评估报告已保存: models/{model_type}_evaluation_report_chinese.csv")
        
    except Exception as e:
        logging.error(f"模型评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_model_performance() 