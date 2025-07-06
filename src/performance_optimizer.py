#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 性能优化器 - 将准确率从77%提升到80%+
"""

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
import os
import json
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """性能优化器 - 目标：77% → 80%"""
    
    def __init__(self):
        self.base_models = {}
        self.optimization_results = {}
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        """加载训练数据"""
        try:
            features = np.load('data/features.npy')
            labels_df = pd.read_csv('data/processed/emotion_labels.csv')
            
            # 重塑特征
            if len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)
            
            logger.info(f"📊 数据加载完成 - 特征: {features.shape}")
            return features, labels_df['emotion'].values
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            return None, None
    
    def create_advanced_ensemble(self, X_train, y_train):
        """创建高级集成模型"""
        logger.info("🚀 创建高级集成模型...")
        
        # 1. 基础强分类器
        base_models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'svm': SVC(
                C=10,
                gamma='scale',
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=1000,
                learning_rate_init=0.001,
                alpha=0.01,
                random_state=42
            )
        }
        
        # 2. 创建Stacking集成
        stacking_classifier = StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=LogisticRegression(
                C=1.0,
                class_weight='balanced',
                random_state=42
            ),
            cv=5,
            n_jobs=1  # 修复Windows编码问题
        )
        
        # 3. 训练模型
        stacking_classifier.fit(X_train, y_train)
        
        # 4. 置信度校准
        calibrated_classifier = CalibratedClassifierCV(
            stacking_classifier,
            method='isotonic',
            cv=3
        )
        calibrated_classifier.fit(X_train, y_train)
        
        return calibrated_classifier
    
    def optimize_features(self, X, y, k=3000):
        """特征优化"""
        logger.info(f"🔧 优化特征选择 (选择前{k}个最重要特征)")
        
        # 使用F检验选择最重要的特征
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        logger.info(f"   特征维度: {X.shape[1]} → {X_selected.shape[1]}")
        return X_selected, selector
    
    def hyperparameter_tuning(self, model, X_train, y_train):
        """超参数调优"""
        logger.info("⚙️ 执行超参数调优...")
        
        # 为Stacking模型定义参数网格
        param_grid = {
            'final_estimator__C': [0.1, 1.0, 10.0],
            'final_estimator__solver': ['liblinear', 'lbfgs']
        }
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,  # 减少CV折数以加快速度
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"   最佳参数: {grid_search.best_params_}")
        logger.info(f"   最佳分数: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def data_augmentation_analysis(self, X, y):
        """数据增强分析"""
        logger.info("📈 分析数据分布和增强需求...")
        
        unique, counts = np.unique(y, return_counts=True)
        emotion_distribution = dict(zip(unique, counts))
        
        logger.info("   当前数据分布:")
        for emotion, count in emotion_distribution.items():
            logger.info(f"     {emotion}: {count}个样本")
        
        # 计算需要增强的类别
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        logger.info(f"   类别不平衡比例: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            logger.info("   🎯 建议进行数据平衡处理")
            return True
        
        return False
    
    def optimize_model(self):
        """主优化流程"""
        logger.info("🎯 开始性能优化 - 目标: 77% → 80%+")
        logger.info("=" * 60)
        
        # 1. 加载数据
        X, y = self.load_data()
        if X is None:
            return None
        
        # 2. 数据分析
        needs_balancing = self.data_augmentation_analysis(X, y)
        
        # 3. 数据分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. 特征优化
        X_train_optimized, feature_selector = self.optimize_features(X_train, y_train)
        X_test_optimized = feature_selector.transform(X_test)
        
        # 5. 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_optimized)
        X_test_scaled = scaler.transform(X_test_optimized)
        
        # 6. 创建高级集成模型
        ensemble_model = self.create_advanced_ensemble(X_train_scaled, y_train)
        
        # 7. 超参数调优
        if hasattr(ensemble_model, 'base_estimator'):
            optimized_model = self.hyperparameter_tuning(
                ensemble_model.base_estimator, X_train_scaled, y_train
            )
            # 重新校准
            final_model = CalibratedClassifierCV(optimized_model, method='isotonic', cv=3)
            final_model.fit(X_train_scaled, y_train)
        else:
            final_model = ensemble_model
        
        # 8. 评估性能
        y_pred = final_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # 9. 交叉验证
        cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5)
        cv_accuracy = cv_scores.mean()
        
        logger.info(f"\n🎯 优化结果:")
        logger.info(f"   测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info(f"   交叉验证准确率: {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
        logger.info(f"   性能提升: {'+' if test_accuracy > 0.77 else ''}{(test_accuracy - 0.77)*100:.2f}%")
        
        # 10. 保存优化结果
        if test_accuracy > 0.77:  # 如果性能有提升
            self.save_optimized_model(
                final_model, scaler, feature_selector, 
                test_accuracy, cv_accuracy
            )
            
            # 详细分类报告
            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info(f"\n📋 详细分类报告:")
            for emotion in np.unique(y):
                if emotion in report:
                    metrics = report[emotion]
                    logger.info(f"   {emotion}: P={metrics['precision']:.3f}, "
                              f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return test_accuracy
    
    def save_optimized_model(self, model, scaler, feature_selector, test_acc, cv_acc):
        """保存优化后的模型"""
        logger.info("💾 保存优化后的模型...")
        
        model_dir = 'models/optimized_enhanced'
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型组件
        joblib.dump(model, f'{model_dir}/enhanced_model.pkl')
        joblib.dump(scaler, f'{model_dir}/scaler.pkl')
        joblib.dump(feature_selector, f'{model_dir}/feature_selector.pkl')
        
        # 保存性能报告
        performance_report = {
            'optimization_date': datetime.now().isoformat(),
            'test_accuracy': float(test_acc),
            'cv_accuracy': float(cv_acc),
            'improvement_over_77pct': float((test_acc - 0.77) * 100),
            'model_type': 'advanced_stacking_ensemble',
            'feature_count': feature_selector.k,
            'calibration': 'isotonic'
        }
        
        with open(f'{model_dir}/performance_report.json', 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   模型已保存到: {model_dir}/")
        logger.info(f"   准确率提升: {performance_report['improvement_over_77pct']:+.2f}%")
        
        return model_dir

def main():
    """主函数"""
    print("🎯 音乐情感识别性能优化器")
    print("目标: 将准确率从77%提升到80%+")
    print("=" * 50)
    
    optimizer = PerformanceOptimizer()
    final_accuracy = optimizer.optimize_model()
    
    if final_accuracy and final_accuracy >= 0.80:
        print(f"\n🎉 优化成功！达到目标准确率: {final_accuracy*100:.2f}%")
        print("✅ 系统现在可以达到80%+的预测准确率！")
    elif final_accuracy and final_accuracy > 0.77:
        print(f"\n📈 优化有效！准确率提升至: {final_accuracy*100:.2f}%")
        print(f"   距离80%目标还差: {(0.80 - final_accuracy)*100:.2f}%")
        print("💡 建议继续收集更多训练数据或尝试深度学习方法")
    else:
        print(f"\n⚠️ 优化效果有限，当前准确率: {final_accuracy*100:.2f}%")
        print("💡 建议分析数据质量或尝试不同的特征工程方法")

if __name__ == "__main__":
    main() 