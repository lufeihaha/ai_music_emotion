#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def improved_ensemble_learning():
    """Improved ensemble learning with better model selection and feature engineering"""
    print("🚀 改进的集成学习测试")
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
        
        # 标签编码
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        
        # 特征选择 - 选择最重要的特征
        selector = SelectKBest(score_func=f_classif, k=min(2000, features.shape[1]))
        X_selected = selector.fit_transform(features, y_encoded)
        
        print(f"🔍 特征选择完成 - 从 {features.shape[1]} 维降至 {X_selected.shape[1]} 维")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"📈 训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")
        
        # 定义优化的基础模型
        base_models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.15,
                max_depth=8,
                subsample=0.8,
                random_state=42
            ),
            'svm': SVC(
                C=10,
                gamma='scale',
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'lr': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=300,
                random_state=42
            )
        }
        
        # 训练和评估单个模型
        individual_scores = {}
        trained_models = {}
        
        print("\n🔧 训练单个模型...")
        for name, model in base_models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            individual_scores[name] = score
            trained_models[name] = model
            print(f"  {name}: {score:.4f}")
        
        # 选择最佳的3个模型进行集成
        best_models = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n🏆 选择最佳3个模型: {[name for name, score in best_models]}")
        
        # 创建集成模型
        ensemble_estimators = [(name, trained_models[name]) for name, score in best_models]
        
        # 尝试不同的集成策略
        voting_strategies = ['soft', 'hard']
        best_ensemble_score = 0
        best_strategy = None
        
        for strategy in voting_strategies:
            if strategy == 'hard':
                # 对于硬投票，需要重新训练不支持概率的模型
                ensemble_estimators_hard = []
                for name, model in ensemble_estimators:
                    if name == 'svm':
                        # 使用不支持概率的SVM
                        svm_hard = SVC(C=10, gamma='scale', kernel='rbf', random_state=42)
                        svm_hard.fit(X_train_scaled, y_train)
                        ensemble_estimators_hard.append((name, svm_hard))
                    else:
                        ensemble_estimators_hard.append((name, model))
                
                ensemble = VotingClassifier(
                    estimators=ensemble_estimators_hard,
                    voting='hard',
                    n_jobs=1
                )
            else:
                ensemble = VotingClassifier(
                    estimators=ensemble_estimators,
                    voting='soft',
                    n_jobs=1
                )
            
            ensemble.fit(X_train_scaled, y_train)
            y_pred_ensemble = ensemble.predict(X_test_scaled)
            ensemble_score = accuracy_score(y_test, y_pred_ensemble)
            
            print(f"  {strategy}投票集成: {ensemble_score:.4f}")
            
            if ensemble_score > best_ensemble_score:
                best_ensemble_score = ensemble_score
                best_strategy = strategy
                best_ensemble = ensemble
        
        # 结果分析
        print(f"\n📊 最终结果:")
        print(f"  最佳单个模型: {best_models[0][0]} = {best_models[0][1]:.4f}")
        print(f"  最佳集成模型: {best_strategy}投票 = {best_ensemble_score:.4f}")
        
        improvement = best_ensemble_score - best_models[0][1]
        improvement_pct = (improvement / best_models[0][1]) * 100
        
        print(f"  改进效果: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        if improvement > 0:
            print("✅ 集成学习成功提升了模型性能!")
            
            # 保存改进的模型
            model_dir = 'models/improved_ensemble'
            import os
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(best_ensemble, f'{model_dir}/ensemble_model.pkl')
            joblib.dump(scaler, f'{model_dir}/scaler.pkl')
            joblib.dump(label_encoder, f'{model_dir}/label_encoder.pkl')
            joblib.dump(selector, f'{model_dir}/feature_selector.pkl')
            
            print(f"💾 模型已保存到 {model_dir}/")
            
            # 详细分类报告
            y_pred_final = best_ensemble.predict(X_test_scaled)
            report = classification_report(y_test, y_pred_final, 
                                         target_names=label_encoder.classes_,
                                         output_dict=True)
            
            print(f"\n📋 详细分类报告:")
            for emotion in label_encoder.classes_:
                metrics = report[emotion]
                print(f"  {emotion}: 精确率={metrics['precision']:.3f}, "
                      f"召回率={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            
        else:
            print("⚠️ 集成学习未能提升性能，建议:")
            print("  1. 增加更多样化的模型")
            print("  2. 调整特征工程方法")
            print("  3. 尝试不同的集成策略")
        
        return best_ensemble_score > best_models[0][1]
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

if __name__ == "__main__":
    improved_ensemble_learning() 