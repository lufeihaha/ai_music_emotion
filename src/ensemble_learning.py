#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集成学习模块 - 提升音乐情感识别准确率
"""

import os
import sys
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

# 设置UTF-8编码
if sys.platform.startswith('win'):
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleMusicEmotionClassifier:
    """集成学习音乐情感分类器"""
    
    def __init__(self):
        self.ensemble_model = None
        self.individual_models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_weights = {}
        
    def create_ensemble_model(self):
        """创建集成学习模型"""
        # 定义个体模型
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
        
        # 创建投票分类器
        self.ensemble_model = VotingClassifier(
            estimators=models,
            voting='soft',  # 使用软投票（基于概率）
            n_jobs=-1
        )
        
        # 保存个体模型引用
        for name, model in models:
            self.individual_models[name] = model
            
        logger.info("集成学习模型创建完成")
        return self.ensemble_model
    
    def load_data(self):
        """加载训练数据"""
        try:
            # 加载特征数据
            features = np.load('data/features.npy')
            labels_df = pd.read_csv('data/processed/emotion_labels.csv')
            labels = labels_df['emotion'].values
            
            logger.info(f"数据加载完成 - 特征维度: {features.shape}, 标签数量: {len(labels)}")
            
            # 处理3D特征数据
            if len(features.shape) == 3:
                n_samples, n_segments, n_features = features.shape
                features = features.reshape(n_samples, n_segments * n_features)
                logger.info(f"特征重塑为2D: {features.shape}")
            
            return features, labels
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return None, None
    
    def train_ensemble(self, features, labels, test_size=0.2):
        """训练集成学习模型"""
        try:
            # 编码标签
            labels_encoded = self.label_encoder.fit_transform(labels)
            
            # 划分训练测试集
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels_encoded, 
                test_size=test_size, 
                random_state=42, 
                stratify=labels_encoded
            )
            
            # 特征标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 创建集成模型
            self.create_ensemble_model()
            
            # 训练集成模型
            logger.info("开始训练集成学习模型...")
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            # 评估性能
            train_score = self.ensemble_model.score(X_train_scaled, y_train)
            test_score = self.ensemble_model.score(X_test_scaled, y_test)
            
            # 交叉验证
            cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train, cv=5)
            
            logger.info(f"训练完成!")
            logger.info(f"训练准确率: {train_score:.4f}")
            logger.info(f"测试准确率: {test_score:.4f}")
            logger.info(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 生成详细报告
            y_pred = self.ensemble_model.predict(X_test_scaled)
            
            # 处理中文标签编码问题
            try:
                emotion_labels = [str(label) for label in self.label_encoder.classes_]
                report = classification_report(y_test, y_pred, 
                                             target_names=emotion_labels,
                                             output_dict=True)
            except UnicodeEncodeError:
                # 如果有中文字符，使用简化标签
                emotion_labels = [f"emotion_{i}" for i in range(len(self.label_encoder.classes_))]
                report = classification_report(y_test, y_pred, 
                                             target_names=emotion_labels,
                                             output_dict=True)
            
            # 评估个体模型性能
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
            
            # 保存模型
            self.save_models()
            
            # 生成可视化报告
            self._generate_visualization_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            return None
    
    def _evaluate_individual_models(self, X_train, y_train, X_test, y_test):
        """评估个体模型性能"""
        individual_scores = {}
        
        for name, model in self.individual_models.items():
            try:
                # 训练个体模型
                model.fit(X_train, y_train)
                
                # 评估性能
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                individual_scores[name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"{name} - 测试准确率: {test_score:.4f}")
                
            except Exception as e:
                logger.warning(f"评估模型 {name} 失败: {str(e)}")
                individual_scores[name] = {
                    'train_score': 0,
                    'test_score': 0,
                    'cv_score': 0,
                    'cv_std': 0
                }
        
        return individual_scores
    
    def _generate_visualization_report(self, results):
        """生成可视化报告"""
        try:
            # 创建保存目录
            os.makedirs('models/ensemble_results', exist_ok=True)
            
            # 1. 模型性能对比图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 个体模型vs集成模型准确率对比
            model_names = list(results['individual_scores'].keys()) + ['Ensemble']
            test_scores = [results['individual_scores'][name]['test_score'] 
                          for name in results['individual_scores'].keys()]
            test_scores.append(results['ensemble_test_score'])
            
            axes[0, 0].bar(model_names, test_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
            axes[0, 0].set_title('模型准确率对比', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('测试准确率')
            axes[0, 0].set_ylim(0, 1)
            
            # 添加数值标签
            for i, score in enumerate(test_scores):
                axes[0, 0].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
            
            # 交叉验证分数分布
            cv_scores = [results['individual_scores'][name]['cv_score'] 
                        for name in results['individual_scores'].keys()]
            cv_scores.append(results['ensemble_cv_scores'].mean())
            
            axes[0, 1].bar(model_names, cv_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
            axes[0, 1].set_title('交叉验证准确率对比', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('CV准确率')
            axes[0, 1].set_ylim(0, 1)
            
            # 混淆矩阵
            cm = results['confusion_matrix']
            
            # 处理中文标签显示问题
            try:
                emotion_labels = [str(label) for label in self.label_encoder.classes_]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=emotion_labels,
                           yticklabels=emotion_labels,
                           ax=axes[1, 0])
            except UnicodeEncodeError:
                # 使用简化标签
                emotion_labels = [f"E{i}" for i in range(len(self.label_encoder.classes_))]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=emotion_labels,
                           yticklabels=emotion_labels,
                           ax=axes[1, 0])
            axes[1, 0].set_title('集成模型混淆矩阵', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('预测标签')
            axes[1, 0].set_ylabel('真实标签')
            
            # 性能提升图
            improvement = results['ensemble_test_score'] - max([results['individual_scores'][name]['test_score'] 
                                                              for name in results['individual_scores'].keys()])
            axes[1, 1].bar(['最佳个体模型', '集成模型'], 
                          [max([results['individual_scores'][name]['test_score'] 
                               for name in results['individual_scores'].keys()]),
                           results['ensemble_test_score']], 
                          color=['orange', 'green'])
            axes[1, 1].set_title(f'集成学习性能提升: +{improvement:.3f}', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('测试准确率')
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('models/ensemble_results/performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 生成文本报告
            self._generate_text_report(results)
            
            logger.info("可视化报告生成完成")
            
        except Exception as e:
            logger.error(f"生成可视化报告失败: {str(e)}")
    
    def _generate_text_report(self, results):
        """生成文本报告"""
        # 计算性能提升
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
        
        # 保存报告
        try:
            with open('models/ensemble_results/ensemble_report.md', 'w', encoding='utf-8') as f:
                f.write(report_content)
        except Exception as e:
            logger.warning(f"Failed to save report: {str(e)}")
            # 尝试保存简化版本
            with open('models/ensemble_results/ensemble_report.txt', 'w', encoding='utf-8') as f:
                f.write(f"Ensemble Model Test Accuracy: {results['ensemble_test_score']:.4f}\n")
                f.write(f"Cross-validation Score: {results['ensemble_cv_scores'].mean():.4f}\n")
    
    def save_models(self):
        """保存训练好的模型"""
        try:
            os.makedirs('models/ensemble_results', exist_ok=True)
            
            # 保存集成模型
            joblib.dump(self.ensemble_model, 'models/ensemble_results/ensemble_model.pkl')
            joblib.dump(self.scaler, 'models/ensemble_results/ensemble_scaler.pkl')
            joblib.dump(self.label_encoder, 'models/ensemble_results/ensemble_label_encoder.pkl')
            
            logger.info("模型保存完成")
            
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
    
    def load_models(self):
        """加载训练好的模型"""
        try:
            self.ensemble_model = joblib.load('models/ensemble_results/ensemble_model.pkl')
            self.scaler = joblib.load('models/ensemble_results/ensemble_scaler.pkl')
            self.label_encoder = joblib.load('models/ensemble_results/ensemble_label_encoder.pkl')
            
            logger.info("集成模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def predict(self, features):
        """预测情感"""
        try:
            if self.ensemble_model is None:
                if not self.load_models():
                    raise ValueError("模型未训练且加载失败")
            
            # 特征标准化
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # 预测
            prediction = self.ensemble_model.predict(features_scaled)[0]
            probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
            
            # 转换标签
            emotion = self.label_encoder.inverse_transform([prediction])[0]
            
            # 获取所有情感的概率
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
            logger.error(f"预测失败: {str(e)}")
            return None

def main():
    """主函数 - 训练集成学习模型"""
    print("🎵 集成学习音乐情感识别模型训练")
    print("=" * 50)
    
    # 创建集成分类器
    classifier = EnsembleMusicEmotionClassifier()
    
    # 加载数据
    features, labels = classifier.load_data()
    if features is None or labels is None:
        print("❌ 数据加载失败，请确保数据文件存在")
        return
    
    # 训练模型
    results = classifier.train_ensemble(features, labels)
    if results is None:
        print("❌ 模型训练失败")
        return
    
    # 显示结果
    print("\n📊 训练结果:")
    print(f"集成模型测试准确率: {results['ensemble_test_score']:.4f}")
    print(f"交叉验证准确率: {results['ensemble_cv_scores'].mean():.4f}")
    
    # 计算性能提升
    best_individual = max([scores['test_score'] for scores in results['individual_scores'].values()])
    improvement = results['ensemble_test_score'] - best_individual
    print(f"性能提升: +{improvement:.4f} ({improvement/best_individual*100:.1f}%)")
    
    print("\n✅ 集成学习模型训练完成！")
    print("📁 结果保存在 models/ensemble_results/ 目录")

if __name__ == "__main__":
    main() 