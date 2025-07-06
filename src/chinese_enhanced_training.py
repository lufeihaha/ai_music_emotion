#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结合中文数据集的增强训练脚本
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from chinese_dataset_downloader import ChineseMusicDatasetDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseEnhancedTrainer:
    """结合中文数据集的增强训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.chinese_downloader = ChineseMusicDatasetDownloader()
        
        # 模型配置
        self.model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'name': '随机森林'
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42
                ),
                'name': '梯度提升'
            },
            'SVM': {
                'model': SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=42,
                    probability=True
                ),
                'name': '支持向量机'
            },
            'MLP': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42
                ),
                'name': '多层感知机'
            }
        }
    
    def prepare_enhanced_dataset(self) -> tuple:
        """
        准备增强后的数据集（原始数据 + 中文数据集）
        
        Returns:
            tuple: (X, y, feature_names)
        """
        try:
            logger.info("开始准备增强数据集...")
            
            # 1. 加载原始特征数据
            original_features = self._load_original_features()
            original_labels = self._load_original_labels()
            
            logger.info(f"原始数据集: {len(original_features)} 个样本")
            
            # 2. 下载并处理中文数据集
            chinese_data = self._prepare_chinese_data()
            
            if chinese_data is not None and len(chinese_data) > 0:
                logger.info(f"中文数据集: {len(chinese_data)} 个样本")
                
                # 3. 合并数据集
                X, y = self._merge_datasets(original_features, original_labels, chinese_data)
                logger.info(f"合并后数据集: {len(X)} 个样本")
            else:
                logger.warning("中文数据集不可用，使用原始数据集")
                X = original_features
                y = original_labels
            
            # 4. 数据预处理
            X_processed, y_processed = self._preprocess_data(X, y)
            
            return X_processed, y_processed, self._get_feature_names()
            
        except Exception as e:
            logger.error(f"准备数据集失败: {str(e)}")
            raise
    
    def _load_original_features(self) -> np.ndarray:
        """加载原始特征数据"""
        features_file = "data/features.npy"
        if os.path.exists(features_file):
            features = np.load(features_file)
            if len(features.shape) == 3:
                # 如果是3D数组，重塑为2D
                n_samples, n_segments, n_features = features.shape
                features = features.reshape(n_samples, n_segments * n_features)
            return features
        else:
            raise FileNotFoundError(f"特征文件不存在: {features_file}")
    
    def _load_original_labels(self) -> np.ndarray:
        """加载原始标签数据"""
        labels_file = "data/processed/emotion_labels.csv"
        if os.path.exists(labels_file):
            df = pd.read_csv(labels_file)
            return df['emotion'].values
        else:
            raise FileNotFoundError(f"标签文件不存在: {labels_file}")
    
    def _prepare_chinese_data(self) -> pd.DataFrame:
        """准备中文数据集"""
        try:
            # 检查是否已有中文数据集
            chinese_file = "data/chinese_datasets/chinese_music_emotions.csv"
            
            if not os.path.exists(chinese_file):
                logger.info("中文数据集不存在，开始下载...")
                
                # 下载中文数据集
                if self.chinese_downloader.download_all_datasets():
                    if self.chinese_downloader.integrate_chinese_datasets():
                        logger.info("中文数据集下载和整合完成")
                    else:
                        logger.warning("中文数据集整合失败")
                        return None
                else:
                    logger.warning("中文数据集下载失败")
                    return None
            
            # 读取中文数据集
            if os.path.exists(chinese_file):
                chinese_df = pd.read_csv(chinese_file)
                logger.info(f"成功加载中文数据集: {len(chinese_df)} 条记录")
                return chinese_df
            else:
                logger.warning("中文数据集文件不存在")
                return None
                
        except Exception as e:
            logger.error(f"准备中文数据失败: {str(e)}")
            return None
    
    def _merge_datasets(self, original_features: np.ndarray, original_labels: np.ndarray, 
                       chinese_data: pd.DataFrame) -> tuple:
        """
        合并原始数据集和中文数据集
        
        Args:
            original_features: 原始特征
            original_labels: 原始标签
            chinese_data: 中文数据集
            
        Returns:
            tuple: (合并后的特征, 合并后的标签)
        """
        try:
            # 为中文数据生成伪特征（基于情感统计特征）
            chinese_features = self._generate_pseudo_features(chinese_data)
            chinese_labels = chinese_data['mapped_emotion'].values
            
            # 合并特征和标签
            X_combined = np.vstack([original_features, chinese_features])
            y_combined = np.hstack([original_labels, chinese_labels])
            
            logger.info(f"数据集合并完成: {len(X_combined)} 个样本")
            
            return X_combined, y_combined
            
        except Exception as e:
            logger.error(f"合并数据集失败: {str(e)}")
            return original_features, original_labels
    
    def _generate_pseudo_features(self, chinese_data: pd.DataFrame) -> np.ndarray:
        """
        为中文数据生成伪特征
        
        Args:
            chinese_data: 中文数据集
            
        Returns:
            np.ndarray: 生成的伪特征
        """
        n_samples = len(chinese_data)
        n_features = 4864  # 与原始特征维度保持一致
        
        # 基于arousal和valence生成特征
        pseudo_features = np.zeros((n_samples, n_features))
        
        for i, row in chinese_data.iterrows():
            arousal = row.get('arousal', 0.5)
            valence = row.get('valence', 0.5)
            emotion = row.get('mapped_emotion', 'calm')
            
            # 生成基于情感特征的伪特征
            feature_vector = self._create_emotion_based_features(arousal, valence, emotion)
            
            # 确保特征向量长度正确
            if len(feature_vector) < n_features:
                # 填充到所需长度
                feature_vector = np.pad(feature_vector, (0, n_features - len(feature_vector)))
            elif len(feature_vector) > n_features:
                # 截断到所需长度
                feature_vector = feature_vector[:n_features]
            
            pseudo_features[i] = feature_vector
        
        return pseudo_features
    
    def _create_emotion_based_features(self, arousal: float, valence: float, emotion: str) -> np.ndarray:
        """
        基于情感创建特征向量
        
        Args:
            arousal: 唤醒度
            valence: 效价
            emotion: 情感类别
            
        Returns:
            np.ndarray: 特征向量
        """
        # 情感模板特征
        emotion_templates = {
            'happy': np.array([0.8, 0.9, 0.7, 0.6, 0.8]),
            'calm': np.array([0.3, 0.7, 0.8, 0.9, 0.4]),
            'energetic': np.array([0.9, 0.5, 0.6, 0.4, 0.9]),
            'melancholic': np.array([0.2, 0.3, 0.4, 0.8, 0.3])
        }
        
        # 获取基础模板
        base_template = emotion_templates.get(emotion, emotion_templates['calm'])
        
        # 基于arousal和valence调整
        arousal_factor = arousal if arousal <= 1 else arousal / 9
        valence_factor = valence if valence <= 1 else valence / 9
        
        # 生成特征向量
        feature_vector = []
        
        # 基础特征（模拟MFCC）
        for i in range(13):
            base_val = base_template[i % len(base_template)]
            noise = np.random.normal(0, 0.1)
            feature_vector.append(base_val * arousal_factor + noise)
        
        # 频谱特征
        for i in range(20):
            spectral_val = valence_factor * np.random.uniform(0.3, 0.8)
            feature_vector.append(spectral_val)
        
        # 色度特征
        for i in range(12):
            chroma_val = (arousal_factor + valence_factor) / 2 * np.random.uniform(0.2, 0.7)
            feature_vector.append(chroma_val)
        
        # 其他特征（零交叉率、RMS等）
        for i in range(10):
            other_val = arousal_factor * np.random.uniform(0.1, 0.9)
            feature_vector.append(other_val)
        
        return np.array(feature_vector)
    
    def _preprocess_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        数据预处理
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            tuple: (预处理后的特征, 编码后的标签)
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"数据预处理完成: {X_scaled.shape[0]} 样本, {X_scaled.shape[1]} 特征")
        logger.info(f"情感类别: {list(self.label_encoder.classes_)}")
        
        return X_scaled, y_encoded
    
    def _get_feature_names(self) -> list:
        """获取特征名称"""
        feature_names = []
        
        # MFCC特征
        for i in range(13):
            feature_names.append(f"mfcc_{i}")
        
        # 频谱特征
        for i in range(20):
            feature_names.append(f"spectral_{i}")
        
        # 色度特征
        for i in range(12):
            feature_names.append(f"chroma_{i}")
        
        # 其他特征
        other_features = ['zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth', 
                         'spectral_rolloff', 'rms_energy', 'tempo', 'spectral_contrast',
                         'tonnetz', 'spectral_flatness', 'spectral_slope']
        feature_names.extend(other_features)
        
        return feature_names
    
    def train_enhanced_models(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        训练增强模型
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            dict: 训练结果
        """
        logger.info("开始训练增强模型...")
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"训练 {config['name']} 模型...")
            
            try:
                # 训练模型
                model = config['model']
                model.fit(X_train, y_train)
                
                # 评估模型
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # 交叉验证
                cv_scores = cross_val_score(model, X, y, cv=5)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 保存模型
                model_path = f"models/enhanced_{model_name.lower()}_model.pkl"
                joblib.dump(model, model_path)
                
                results[model_name] = {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'y_test': y_test,
                    'model_path': model_path
                }
                
                logger.info(f"{config['name']} - 训练: {train_score:.4f}, 测试: {test_score:.4f}, CV: {cv_scores.mean():.4f}(±{cv_scores.std():.4f})")
                
            except Exception as e:
                logger.error(f"训练 {config['name']} 失败: {str(e)}")
        
        # 保存预处理器
        joblib.dump(self.scaler, "models/enhanced_scaler.pkl")
        joblib.dump(self.label_encoder, "models/enhanced_label_encoder.pkl")
        
        return results
    
    def create_comparison_report(self, results: dict):
        """
        创建模型比较报告
        
        Args:
            results: 训练结果
        """
        logger.info("生成模型比较报告...")
        
        # 创建结果目录
        os.makedirs("models/enhanced_results", exist_ok=True)
        
        # 1. 性能比较图
        self._plot_model_comparison(results)
        
        # 2. 混淆矩阵
        self._plot_confusion_matrices(results)
        
        # 3. 详细报告
        self._generate_detailed_report(results)
    
    def _plot_model_comparison(self, results: dict):
        """绘制模型性能比较图"""
        model_names = []
        train_scores = []
        test_scores = []
        cv_scores = []
        
        for model_name, result in results.items():
            model_names.append(self.model_configs[model_name]['name'])
            train_scores.append(result['train_score'])
            test_scores.append(result['test_score'])
            cv_scores.append(result['cv_mean'])
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, train_scores, width, label='训练准确率', alpha=0.8)
        plt.bar(x, test_scores, width, label='测试准确率', alpha=0.8)
        plt.bar(x + width, cv_scores, width, label='交叉验证准确率', alpha=0.8)
        
        plt.xlabel('模型')
        plt.ylabel('准确率')
        plt.title('增强模型性能比较')
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("models/enhanced_results/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("模型比较图已保存")
    
    def _plot_confusion_matrices(self, results: dict):
        """绘制混淆矩阵"""
        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        emotion_labels = self.label_encoder.classes_
        
        for i, (model_name, result) in enumerate(results.items()):
            if i >= 4:  # 最多显示4个模型
                break
                
            cm = confusion_matrix(result['y_test'], result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=emotion_labels, yticklabels=emotion_labels,
                       ax=axes[i])
            axes[i].set_title(f'{self.model_configs[model_name]["name"]} 混淆矩阵')
            axes[i].set_xlabel('预测标签')
            axes[i].set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig("models/enhanced_results/confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("混淆矩阵已保存")
    
    def _generate_detailed_report(self, results: dict):
        """生成详细报告"""
        report_content = []
        report_content.append("# 中文数据集增强训练报告\n")
        report_content.append(f"生成时间: {pd.Timestamp.now()}\n\n")
        
        report_content.append("## 模型性能总结\n")
        report_content.append("| 模型 | 训练准确率 | 测试准确率 | 交叉验证均值 | 交叉验证标准差 |\n")
        report_content.append("|------|------------|------------|--------------|----------------|\n")
        
        for model_name, result in results.items():
            chinese_name = self.model_configs[model_name]['name']
            report_content.append(
                f"| {chinese_name} | {result['train_score']:.4f} | "
                f"{result['test_score']:.4f} | {result['cv_mean']:.4f} | "
                f"{result['cv_std']:.4f} |\n"
            )
        
        report_content.append("\n## 详细分类报告\n")
        
        for model_name, result in results.items():
            chinese_name = self.model_configs[model_name]['name']
            report_content.append(f"\n### {chinese_name}\n")
            
            # 分类报告
            class_report = classification_report(
                result['y_test'], result['predictions'],
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            report_content.append("| 情感 | 精确率 | 召回率 | F1分数 | 支持数 |\n")
            report_content.append("|------|--------|--------|--------|--------|\n")
            
            for emotion in self.label_encoder.classes_:
                metrics = class_report[emotion]
                report_content.append(
                    f"| {emotion} | {metrics['precision']:.4f} | "
                    f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
                    f"{int(metrics['support'])} |\n"
                )
        
        # 保存报告
        with open("models/enhanced_results/training_report.md", "w", encoding='utf-8') as f:
            f.writelines(report_content)
        
        logger.info("详细报告已保存")

def main():
    """主函数"""
    print("🎵 中文数据集增强训练")
    print("=" * 50)
    
    trainer = ChineseEnhancedTrainer()
    
    try:
        # 准备增强数据集
        print("📊 准备增强数据集...")
        X, y, feature_names = trainer.prepare_enhanced_dataset()
        
        # 训练模型
        print("🚀 开始训练增强模型...")
        results = trainer.train_enhanced_models(X, y)
        
        if results:
            print("✅ 模型训练完成!")
            
            # 生成报告
            print("📋 生成比较报告...")
            trainer.create_comparison_report(results)
            
            print("🎯 训练结果:")
            for model_name, result in results.items():
                chinese_name = trainer.model_configs[model_name]['name']
                print(f"  {chinese_name}: 测试准确率 {result['test_score']:.4f}")
            
            print(f"\n📁 结果保存在: models/enhanced_results/")
        else:
            print("❌ 模型训练失败")
            
    except Exception as e:
        print(f"❌ 训练过程出错: {str(e)}")
        logger.error(f"训练失败: {str(e)}")

if __name__ == "__main__":
    main() 