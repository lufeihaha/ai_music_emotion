#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """音乐情感识别模型优化器"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif, k=2000)
        self.optimization_history = []
        
    def load_data(self):
        """加载数据"""
        try:
            features = np.load('data/features.npy')
            labels_df = pd.read_csv('data/processed/emotion_labels.csv')
            labels = labels_df['emotion'].values
            
            # 重塑特征
            if len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)
            
            logger.info(f"数据加载完成 - 特征: {features.shape}, 标签: {len(labels)}")
            return features, labels
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return None, None
    
    def analyze_confidence_issues(self):
        """分析置信度问题"""
        try:
            # 加载当前最佳模型
            model_path = 'models/improved_ensemble'
            ensemble_model = joblib.load(f'{model_path}/ensemble_model.pkl')
            scaler = joblib.load(f'{model_path}/scaler.pkl')
            label_encoder = joblib.load(f'{model_path}/label_encoder.pkl')
            feature_selector = joblib.load(f'{model_path}/feature_selector.pkl')
            
            # 加载测试数据
            features, labels = self.load_data()
            if features is None:
                return None
            
            # 预处理
            y_encoded = label_encoder.transform(labels)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
            
            X_test_selected = feature_selector.transform(X_test)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # 获取预测概率
            probabilities = ensemble_model.predict_proba(X_test_scaled)
            predictions = ensemble_model.predict(X_test_scaled)
            
            # 分析置信度分布
            max_probs = np.max(probabilities, axis=1)
            
            confidence_analysis = {
                'mean_confidence': np.mean(max_probs),
                'std_confidence': np.std(max_probs),
                'low_confidence_count': np.sum(max_probs < 0.5),
                'high_confidence_count': np.sum(max_probs > 0.8),
                'accuracy': accuracy_score(y_test, predictions)
            }
            
            logger.info(f"置信度分析结果:")
            logger.info(f"  平均置信度: {confidence_analysis['mean_confidence']:.3f}")
            logger.info(f"  置信度标准差: {confidence_analysis['std_confidence']:.3f}")
            logger.info(f"  低置信度样本 (<0.5): {confidence_analysis['low_confidence_count']}")
            logger.info(f"  高置信度样本 (>0.8): {confidence_analysis['high_confidence_count']}")
            
            # 绘制置信度分布图
            plt.figure(figsize=(10, 6))
            plt.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=0.5, color='red', linestyle='--', label='低置信度阈值')
            plt.axvline(x=0.8, color='green', linestyle='--', label='高置信度阈值')
            plt.xlabel('预测置信度')
            plt.ylabel('样本数量')
            plt.title('模型预测置信度分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('models/visualizations/confidence_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return confidence_analysis
            
        except Exception as e:
            logger.error(f"置信度分析失败: {e}")
            return None
    
    def optimize_hyperparameters(self):
        """超参数优化"""
        try:
            features, labels = self.load_data()
            if features is None:
                return None
            
            # 预处理
            y_encoded = self.label_encoder.fit_transform(labels)
            X_selected = self.feature_selector.fit_transform(features, y_encoded)
            X_scaled = self.scaler.fit_transform(X_selected)
            
            # 定义超参数搜索空间
            param_grids = {
                'RandomForest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'SVM': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly']
                }
            }
            
            # 基础模型
            base_models = {
                'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(probability=True, random_state=42)
            }
            
            optimized_models = {}
            
            for name, model in base_models.items():
                logger.info(f"优化 {name} 超参数...")
                
                # 使用随机搜索以节省时间
                random_search = RandomizedSearchCV(
                    model, param_grids[name], 
                    n_iter=20, cv=3, 
                    scoring='accuracy', 
                    random_state=42, 
                    n_jobs=-1
                )
                
                random_search.fit(X_scaled, y_encoded)
                
                optimized_models[name] = {
                    'model': random_search.best_estimator_,
                    'best_params': random_search.best_params_,
                    'best_score': random_search.best_score_
                }
                
                logger.info(f"{name} 最佳得分: {random_search.best_score_:.4f}")
                logger.info(f"{name} 最佳参数: {random_search.best_params_}")
            
            # 创建优化后的集成模型
            best_models = [(name, data['model']) for name, data in optimized_models.items()]
            
            optimized_ensemble = VotingClassifier(
                estimators=best_models,
                voting='soft'
            )
            
            optimized_ensemble.fit(X_scaled, y_encoded)
            
            # 保存优化后的模型
            os.makedirs('models/optimized_ensemble', exist_ok=True)
            joblib.dump(optimized_ensemble, 'models/optimized_ensemble/ensemble_model.pkl')
            joblib.dump(self.scaler, 'models/optimized_ensemble/scaler.pkl')
            joblib.dump(self.label_encoder, 'models/optimized_ensemble/label_encoder.pkl')
            joblib.dump(self.feature_selector, 'models/optimized_ensemble/feature_selector.pkl')
            
            # 保存优化历史
            optimization_result = {
                'timestamp': datetime.now().isoformat(),
                'optimized_models': {name: data['best_params'] for name, data in optimized_models.items()},
                'best_scores': {name: data['best_score'] for name, data in optimized_models.items()}
            }
            
            with open('models/optimized_ensemble/optimization_history.json', 'w') as f:
                json.dump(optimization_result, f, indent=2)
            
            logger.info("超参数优化完成，模型已保存")
            return optimized_models
            
        except Exception as e:
            logger.error(f"超参数优化失败: {e}")
            return None
    
    def analyze_user_feedback(self):
        """分析用户反馈数据"""
        try:
            # 查找用户反馈文件
            feedback_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if 'feedback' in file.lower() and file.endswith('.json'):
                        feedback_files.append(os.path.join(root, file))
            
            if not feedback_files:
                logger.warning("未找到用户反馈文件")
                return None
            
            # 读取反馈数据
            feedback_data = []
            for file_path in feedback_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            feedback_data.extend(data)
                        else:
                            feedback_data.append(data)
                except Exception as e:
                    logger.warning(f"读取反馈文件失败 {file_path}: {e}")
            
            if not feedback_data:
                logger.warning("未找到有效的反馈数据")
                return None
            
            # 分析反馈数据
            df = pd.DataFrame(feedback_data)
            
            analysis = {
                'total_feedback': len(df),
                'average_rating': df['user_rating'].mean() if 'user_rating' in df.columns else None,
                'emotion_accuracy': {},
                'common_mistakes': {}
            }
            
            if 'predicted_emotion' in df.columns and 'actual_emotion' in df.columns:
                # 计算每种情感的准确率
                for emotion in df['predicted_emotion'].unique():
                    emotion_data = df[df['predicted_emotion'] == emotion]
                    if len(emotion_data) > 0:
                        correct = len(emotion_data[emotion_data['predicted_emotion'] == emotion_data['actual_emotion']])
                        analysis['emotion_accuracy'][emotion] = correct / len(emotion_data)
                
                # 分析常见错误
                mistakes = df[df['predicted_emotion'] != df['actual_emotion']]
                if len(mistakes) > 0:
                    mistake_counts = mistakes.groupby(['predicted_emotion', 'actual_emotion']).size()
                    analysis['common_mistakes'] = mistake_counts.to_dict()
            
            logger.info(f"用户反馈分析结果:")
            logger.info(f"  总反馈数: {analysis['total_feedback']}")
            if analysis['average_rating']:
                logger.info(f"  平均评分: {analysis['average_rating']:.2f}")
            
            # 保存分析结果
            os.makedirs('models/feedback_analysis', exist_ok=True)
            with open('models/feedback_analysis/feedback_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            
            return analysis
            
        except Exception as e:
            logger.error(f"用户反馈分析失败: {e}")
            return None
    
    def create_rejection_threshold_model(self, threshold=0.6):
        """创建带拒绝阈值的模型"""
        try:
            # 加载当前最佳模型
            model_path = 'models/improved_ensemble'
            ensemble_model = joblib.load(f'{model_path}/ensemble_model.pkl')
            scaler = joblib.load(f'{model_path}/scaler.pkl')
            label_encoder = joblib.load(f'{model_path}/label_encoder.pkl')
            feature_selector = joblib.load(f'{model_path}/feature_selector.pkl')
            
            class RejectionModel:
                def __init__(self, model, scaler, label_encoder, feature_selector, threshold):
                    self.model = model
                    self.scaler = scaler
                    self.label_encoder = label_encoder
                    self.feature_selector = feature_selector
                    self.threshold = threshold
                
                def predict_with_rejection(self, X):
                    """带拒绝的预测"""
                    # 预处理
                    X_selected = self.feature_selector.transform(X)
                    X_scaled = self.scaler.transform(X_selected)
                    
                    # 获取预测概率
                    probabilities = self.model.predict_proba(X_scaled)
                    max_probs = np.max(probabilities, axis=1)
                    predictions = self.model.predict(X_scaled)
                    
                    # 应用拒绝阈值
                    results = []
                    for i, (pred, prob) in enumerate(zip(predictions, max_probs)):
                        if prob >= self.threshold:
                            emotion = self.label_encoder.inverse_transform([pred])[0]
                            results.append({
                                'emotion': emotion,
                                'confidence': prob,
                                'rejected': False
                            })
                        else:
                            results.append({
                                'emotion': '不确定',
                                'confidence': prob,
                                'rejected': True
                            })
                    
                    return results
            
            rejection_model = RejectionModel(
                ensemble_model, scaler, label_encoder, feature_selector, threshold
            )
            
            # 保存拒绝模型
            os.makedirs('models/rejection_model', exist_ok=True)
            joblib.dump(rejection_model, 'models/rejection_model/rejection_model.pkl')
            
            logger.info(f"拒绝阈值模型创建完成，阈值: {threshold}")
            return rejection_model
            
        except Exception as e:
            logger.error(f"创建拒绝阈值模型失败: {e}")
            return None
    
    def generate_optimization_report(self):
        """生成优化报告"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'current_best_model': 'improved_ensemble',
                'current_accuracy': 0.6867,
                'optimization_recommendations': []
            }
            
            # 分析置信度
            confidence_analysis = self.analyze_confidence_issues()
            if confidence_analysis:
                report['confidence_analysis'] = confidence_analysis
                
                if confidence_analysis['mean_confidence'] < 0.7:
                    report['optimization_recommendations'].append({
                        'type': 'confidence_improvement',
                        'description': '平均置信度较低，建议增加训练数据或调整模型架构',
                        'priority': 'high'
                    })
            
            # 分析用户反馈
            feedback_analysis = self.analyze_user_feedback()
            if feedback_analysis:
                report['feedback_analysis'] = feedback_analysis
                
                if feedback_analysis['average_rating'] and feedback_analysis['average_rating'] < 3.0:
                    report['optimization_recommendations'].append({
                        'type': 'user_satisfaction',
                        'description': '用户满意度较低，建议基于反馈数据进行模型微调',
                        'priority': 'high'
                    })
            
            # 保存报告
            os.makedirs('models/optimization_reports', exist_ok=True)
            report_path = f'models/optimization_reports/optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"优化报告已保存: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"生成优化报告失败: {e}")
            return None

def main():
    """主函数"""
    print("🎯 音乐情感识别模型优化器")
    print("=" * 50)
    
    optimizer = ModelOptimizer()
    
    try:
        # 1. 分析置信度问题
        print("📊 分析置信度分布...")
        confidence_analysis = optimizer.analyze_confidence_issues()
        
        # 2. 分析用户反馈
        print("👥 分析用户反馈...")
        feedback_analysis = optimizer.analyze_user_feedback()
        
        # 3. 创建拒绝阈值模型
        print("🚫 创建拒绝阈值模型...")
        rejection_model = optimizer.create_rejection_threshold_model(threshold=0.6)
        
        # 4. 生成优化报告
        print("📋 生成优化报告...")
        report = optimizer.generate_optimization_report()
        
        print("\n✅ 模型优化分析完成!")
        print("📁 结果保存在 models/ 相关目录中")
        
        # 5. 可选：超参数优化（较耗时）
        print("\n🔧 是否进行超参数优化？(这可能需要较长时间)")
        print("如需进行，请手动运行: optimizer.optimize_hyperparameters()")
        
    except Exception as e:
        print(f"❌ 优化过程出错: {e}")
        logger.error(f"优化失败: {e}")

if __name__ == "__main__":
    main() 