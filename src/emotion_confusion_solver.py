#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionConfusionSolver:
    """情感混淆问题解决器"""
    
    def __init__(self):
        self.confusion_pairs = [
            ('平静', '怀念'),  # 最常见的混淆
            ('激昂', '怀念'),  # 第二常见的混淆
            ('平静', '浪漫'),  # 其他混淆
            ('平静', '忧郁')
        ]
        self.enhanced_features = {}
        
    def extract_enhanced_features(self, audio_file):
        """提取增强特征来区分混淆的情感"""
        try:
            y, sr = librosa.load(audio_file, duration=30)
            
            features = {}
            
            # 1. 基础特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 2. 情感区分特征
            
            # 节拍和律动特征 (区分平静vs激昂)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            features['beat_strength'] = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
            
            # 音调和和声特征 (区分怀念的音乐特点)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # 调性分析 (怀念歌曲通常是小调)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
            
            # 频谱特征 (区分音色)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = np.mean(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
            # 零交叉率 (区分音频纹理)
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr'] = np.mean(zcr)
            
            # RMS能量 (动态范围)
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # 3. 情感特定特征
            
            # 怀念情感特征 (慢节拍 + 小调 + 低能量)
            features['nostalgia_score'] = self._calculate_nostalgia_score(
                tempo, chroma, rms, spectral_centroid=features['spectral_centroid']
            )
            
            # 平静情感特征 (稳定性 + 低变化)
            features['calmness_score'] = self._calculate_calmness_score(
                rms, spectral_centroids, zcr
            )
            
            # 激昂情感特征 (高能量 + 快节拍)
            features['energy_score'] = self._calculate_energy_score(
                tempo, rms, features['beat_strength']
            )
            
            # 浪漫情感特征 (中等节拍 + 和谐音调)
            features['romantic_score'] = self._calculate_romantic_score(
                tempo, chroma, tonnetz
            )
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None
    
    def _calculate_nostalgia_score(self, tempo, chroma, rms, spectral_centroid):
        """计算怀念情感得分"""
        # 怀念歌曲特征: 慢节拍 + 小调倾向 + 中等能量 + 较低音调
        tempo_score = max(0, 1 - (tempo - 60) / 60)  # 偏爱60-120 BPM
        
        # 小调特征 (基于色度特征的复杂度)
        minor_score = np.std(chroma) / (np.mean(chroma) + 1e-8)
        
        # 能量不太高也不太低
        energy_score = 1 - abs(np.mean(rms) - 0.1) / 0.1
        
        # 音调偏低
        pitch_score = max(0, 1 - (spectral_centroid - 1000) / 2000)
        
        return (tempo_score + minor_score + energy_score + pitch_score) / 4
    
    def _calculate_calmness_score(self, rms, spectral_centroids, zcr):
        """计算平静情感得分"""
        # 平静歌曲特征: 低能量变化 + 稳定频谱 + 低零交叉率
        energy_stability = 1 / (np.std(rms) + 1e-8)
        spectral_stability = 1 / (np.std(spectral_centroids) + 1e-8)
        texture_smoothness = 1 / (np.mean(zcr) + 1e-8)
        
        # 归一化
        energy_stability = min(energy_stability, 10) / 10
        spectral_stability = min(spectral_stability, 1000) / 1000
        texture_smoothness = min(texture_smoothness, 100) / 100
        
        return (energy_stability + spectral_stability + texture_smoothness) / 3
    
    def _calculate_energy_score(self, tempo, rms, beat_strength):
        """计算激昂情感得分"""
        # 激昂歌曲特征: 快节拍 + 高能量 + 强节拍
        tempo_score = min((tempo - 100) / 100, 1) if tempo > 100 else 0
        energy_score = min(np.mean(rms) * 10, 1)
        beat_score = min(beat_strength / 10, 1)
        
        return (tempo_score + energy_score + beat_score) / 3
    
    def _calculate_romantic_score(self, tempo, chroma, tonnetz):
        """计算浪漫情感得分"""
        # 浪漫歌曲特征: 中等节拍 + 和谐音调 + 稳定和弦
        tempo_score = 1 - abs(tempo - 90) / 90  # 偏爱80-100 BPM
        harmony_score = np.mean(chroma)  # 和谐度
        chord_stability = 1 / (np.std(tonnetz) + 1e-8)
        chord_stability = min(chord_stability, 10) / 10
        
        return (tempo_score + harmony_score + chord_stability) / 3
    
    def create_confusion_specific_model(self):
        """创建专门解决混淆问题的模型"""
        try:
            # 加载现有数据
            features = np.load('data/features.npy')
            labels_df = pd.read_csv('data/processed/emotion_labels.csv')
            labels = labels_df['emotion'].values
            
            # 重塑特征
            if len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)
            
            logger.info(f"加载数据: {features.shape}, 标签: {len(labels)}")
            
            # 为混淆的情感对创建特殊的二分类器
            confusion_models = {}
            
            for emotion1, emotion2 in self.confusion_pairs:
                logger.info(f"训练 {emotion1} vs {emotion2} 分类器...")
                
                # 筛选相关数据
                mask = (labels == emotion1) | (labels == emotion2)
                if np.sum(mask) < 10:  # 样本太少，跳过
                    logger.warning(f"样本不足，跳过 {emotion1} vs {emotion2}")
                    continue
                
                X_pair = features[mask]
                y_pair = labels[mask]
                
                # 二分类标签编码
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_pair)
                
                # 训练专门的分类器
                X_train, X_test, y_train, y_test = train_test_split(
                    X_pair, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
                )
                
                # 使用特征选择和增强参数
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 针对混淆问题优化的随机森林
                model = RandomForestClassifier(
                    n_estimators=500,  # 更多树
                    max_depth=15,      # 适中深度
                    min_samples_split=2,  # 更细分
                    min_samples_leaf=1,
                    max_features='sqrt',
                    class_weight='balanced',  # 平衡类别
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train_scaled, y_train)
                
                # 评估性能
                y_pred = model.predict(X_test_scaled)
                accuracy = np.mean(y_pred == y_test)
                
                logger.info(f"{emotion1} vs {emotion2} 准确率: {accuracy:.3f}")
                
                # 保存模型
                confusion_models[f"{emotion1}_vs_{emotion2}"] = {
                    'model': model,
                    'scaler': scaler,
                    'label_encoder': le,
                    'accuracy': accuracy,
                    'emotions': [emotion1, emotion2]
                }
            
            # 保存所有混淆模型
            os.makedirs('models/confusion_solvers', exist_ok=True)
            for name, model_data in confusion_models.items():
                joblib.dump(model_data, f'models/confusion_solvers/{name}.pkl')
            
            logger.info(f"保存了 {len(confusion_models)} 个混淆解决模型")
            return confusion_models
            
        except Exception as e:
            logger.error(f"创建混淆模型失败: {e}")
            return None
    
    def create_enhanced_emotion_classifier(self):
        """创建增强的情感分类器"""
        try:
            # 生成针对混淆问题的合成训练数据
            synthetic_data = self._generate_synthetic_training_data()
            
            if synthetic_data is None:
                logger.error("合成数据生成失败")
                return None
            
            X_synthetic, y_synthetic = synthetic_data
            
            # 加载原始数据
            features = np.load('data/features.npy')
            labels_df = pd.read_csv('data/processed/emotion_labels.csv')
            labels = labels_df['emotion'].values
            
            if len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)
            
            # 合并原始数据和合成数据
            X_combined = np.vstack([features, X_synthetic])
            y_combined = np.hstack([labels, y_synthetic])
            
            logger.info(f"训练数据: 原始 {len(labels)} + 合成 {len(y_synthetic)} = 总计 {len(y_combined)}")
            
            # 编码标签
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_combined)
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练增强模型
            enhanced_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            enhanced_model.fit(X_train_scaled, y_train)
            
            # 评估
            y_pred = enhanced_model.predict(X_test_scaled)
            accuracy = np.mean(y_pred == y_test)
            
            logger.info(f"增强模型准确率: {accuracy:.3f}")
            
            # 生成分类报告
            emotion_names = le.classes_
            report = classification_report(y_test, y_pred, target_names=emotion_names)
            logger.info(f"分类报告:\n{report}")
            
            # 保存模型
            os.makedirs('models/enhanced_emotion', exist_ok=True)
            joblib.dump(enhanced_model, 'models/enhanced_emotion/model.pkl')
            joblib.dump(scaler, 'models/enhanced_emotion/scaler.pkl')
            joblib.dump(le, 'models/enhanced_emotion/label_encoder.pkl')
            
            # 生成混淆矩阵可视化
            self._plot_confusion_matrix(y_test, y_pred, emotion_names)
            
            return {
                'model': enhanced_model,
                'scaler': scaler,
                'label_encoder': le,
                'accuracy': accuracy
            }
            
        except Exception as e:
            logger.error(f"创建增强分类器失败: {e}")
            return None
    
    def _generate_synthetic_training_data(self):
        """生成合成训练数据来解决混淆问题"""
        try:
            # 基于音乐理论生成特征模板
            emotion_templates = {
                '怀念': {
                    'tempo': 75,           # 慢节拍
                    'energy': 0.3,         # 中低能量
                    'spectral_centroid': 1200,  # 偏低音调
                    'nostalgia_score': 0.8,
                    'calmness_score': 0.6,
                    'energy_score': 0.2,
                    'romantic_score': 0.7
                },
                '平静': {
                    'tempo': 85,           # 稍慢节拍
                    'energy': 0.2,         # 低能量
                    'spectral_centroid': 1000,  # 低音调
                    'nostalgia_score': 0.3,
                    'calmness_score': 0.9,
                    'energy_score': 0.1,
                    'romantic_score': 0.4
                },
                '激昂': {
                    'tempo': 130,          # 快节拍
                    'energy': 0.8,         # 高能量
                    'spectral_centroid': 2000,  # 高音调
                    'nostalgia_score': 0.1,
                    'calmness_score': 0.1,
                    'energy_score': 0.9,
                    'romantic_score': 0.2
                },
                '浪漫': {
                    'tempo': 90,           # 中等节拍
                    'energy': 0.4,         # 中等能量
                    'spectral_centroid': 1500,  # 中等音调
                    'nostalgia_score': 0.5,
                    'calmness_score': 0.5,
                    'energy_score': 0.3,
                    'romantic_score': 0.9
                },
                '忧郁': {
                    'tempo': 70,           # 慢节拍
                    'energy': 0.25,        # 低能量
                    'spectral_centroid': 900,   # 很低音调
                    'nostalgia_score': 0.6,
                    'calmness_score': 0.4,
                    'energy_score': 0.1,
                    'romantic_score': 0.3
                }
            }
            
            # 生成合成特征向量
            synthetic_features = []
            synthetic_labels = []
            
            # 为每种情感生成100个合成样本
            for emotion, template in emotion_templates.items():
                for _ in range(100):
                    # 基于模板生成特征，添加随机噪声
                    feature_vector = []
                    
                    # 生成4864维特征向量 (与原始数据一致)
                    for i in range(4864):
                        if i < len(template):
                            # 使用模板值加噪声
                            base_value = list(template.values())[i % len(template)]
                            noise = np.random.normal(0, 0.1)
                            feature_vector.append(base_value + noise)
                        else:
                            # 随机特征
                            feature_vector.append(np.random.normal(0, 0.5))
                    
                    synthetic_features.append(feature_vector)
                    synthetic_labels.append(emotion)
            
            X_synthetic = np.array(synthetic_features)
            y_synthetic = np.array(synthetic_labels)
            
            logger.info(f"生成合成数据: {X_synthetic.shape}")
            return X_synthetic, y_synthetic
            
        except Exception as e:
            logger.error(f"合成数据生成失败: {e}")
            return None
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """绘制混淆矩阵"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('增强情感分类器 - 混淆矩阵')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.tight_layout()
            plt.savefig('models/enhanced_emotion/confusion_matrix.png', dpi=300)
            plt.close()
            
            logger.info("混淆矩阵已保存")
            
        except Exception as e:
            logger.error(f"绘制混淆矩阵失败: {e}")

def main():
    """主函数"""
    print("🔧 情感混淆问题解决器")
    print("=" * 50)
    
    solver = EmotionConfusionSolver()
    
    try:
        # 1. 创建混淆特定模型
        print("📊 创建混淆特定分类器...")
        confusion_models = solver.create_confusion_specific_model()
        
        if confusion_models:
            print(f"✅ 成功创建 {len(confusion_models)} 个混淆解决模型")
        
        # 2. 创建增强情感分类器
        print("🚀 创建增强情感分类器...")
        enhanced_model = solver.create_enhanced_emotion_classifier()
        
        if enhanced_model:
            print(f"✅ 增强模型训练完成，准确率: {enhanced_model['accuracy']:.3f}")
        
        print("\n💡 优化建议:")
        print("  1. 使用增强模型替代原始模型")
        print("  2. 对低置信度预测使用混淆特定模型进行二次判断")
        print("  3. 收集更多'怀念'类型的音乐数据")
        print("  4. 考虑引入歌词分析来区分'怀念'情感")
        
        print(f"\n📁 模型保存位置:")
        print(f"  - 增强模型: models/enhanced_emotion/")
        print(f"  - 混淆解决器: models/confusion_solvers/")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        logger.error(f"主函数执行失败: {e}")

if __name__ == "__main__":
    main() 