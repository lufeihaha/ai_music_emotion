#!/usr/bin/env python3
"""
模拟数据生成器
用于在没有真实音频文件时演示机器学习流程
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple
import logging

class MockDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, config_path: str = "../config.json"):
        """
        初始化模拟数据生成器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.dataset_config = self.config['dataset']
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 随机种子
        np.random.seed(42)
    
    def generate_mock_features(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成模拟特征数据
        
        Args:
            num_samples: 样本数量
            
        Returns:
            特征矩阵、情感标签、标签ID
        """
        self.logger.info(f"生成 {num_samples} 个模拟样本...")
        
        # 特征维度 (根据配置估算)
        feature_dim = 0
        
        # MFCC特征
        if self.config['feature_extraction']['mfcc']['enabled']:
            mfcc_dim = self.config['feature_extraction']['mfcc']['n_mfcc']
            feature_dim += mfcc_dim * 13  # 统计特征数 (mean, std, etc.)
        
        # 频谱特征
        if self.config['feature_extraction']['spectral']['enabled']:
            spectral_features = self.config['feature_extraction']['spectral']['features']
            feature_dim += len(spectral_features) * 13
        
        # Chroma特征
        if self.config['feature_extraction']['chroma']['enabled']:
            chroma_dim = self.config['feature_extraction']['chroma']['n_chroma']
            feature_dim += chroma_dim * 13
        
        # 节奏特征
        if self.config['feature_extraction']['rhythm']['enabled']:
            feature_dim += 3 * 13  # tempo, beat相关特征
        
        # Mel频谱特征
        if self.config['feature_extraction']['mel_spectrogram']['enabled']:
            mel_dim = self.config['feature_extraction']['mel_spectrogram']['n_mels']
            feature_dim += mel_dim * 13
        
        # 如果特征维度仍为0，设置默认值
        if feature_dim == 0:
            feature_dim = 180  # 默认特征维度
        
        self.logger.info(f"特征维度: {feature_dim}")
        
        # 情感映射
        emotions = list(self.dataset_config['emotions'].keys())
        num_emotions = len(emotions)
        
        # 生成特征矩阵
        X = np.random.randn(num_samples, feature_dim)
        
        # 为每种情感创建不同的特征分布
        emotion_labels = []
        label_ids = []
        
        samples_per_emotion = num_samples // num_emotions
        
        for i, emotion in enumerate(emotions):
            # 为每种情感生成特定的特征模式
            start_idx = i * samples_per_emotion
            end_idx = start_idx + samples_per_emotion
            
            if i == len(emotions) - 1:  # 最后一个情感类别包含剩余样本
                end_idx = num_samples
            
            # 为不同情感创建不同的特征偏移
            emotion_offset = np.random.randn(feature_dim) * 2
            X[start_idx:end_idx] += emotion_offset
            
            # 添加情感标签
            emotion_count = end_idx - start_idx
            emotion_labels.extend([emotion] * emotion_count)
            label_ids.extend([i] * emotion_count)
        
        # 打乱数据
        indices = np.random.permutation(num_samples)
        X = X[indices]
        emotion_labels = np.array(emotion_labels)[indices]
        label_ids = np.array(label_ids)[indices]
        
        self.logger.info("模拟数据生成完成")
        self.logger.info(f"情感分布: {dict(zip(*np.unique(emotion_labels, return_counts=True)))}")
        
        return X, emotion_labels, label_ids
    
    def save_mock_data(self, X: np.ndarray, emotions: np.ndarray, label_ids: np.ndarray, output_dir: str = "../data/processed"):
        """
        保存模拟数据
        
        Args:
            X: 特征矩阵
            emotions: 情感标签
            label_ids: 标签ID
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存特征数据
        features_df = pd.DataFrame(X)
        features_df['emotion'] = emotions
        features_df['label_id'] = label_ids
        
        features_path = output_path / "features.csv"
        features_df.to_csv(features_path, index=False)
        self.logger.info(f"保存特征数据到: {features_path}")
        
        # 创建标签映射
        emotions_unique = np.unique(emotions)
        emotion_to_id = {emotion: i for i, emotion in enumerate(emotions_unique)}
        id_to_emotion = {i: emotion for i, emotion in enumerate(emotions_unique)}
        
        mapping_data = {
            'emotion_to_id': emotion_to_id,
            'id_to_emotion': id_to_emotion,
            'emotions': emotions_unique.tolist()
        }
        
        mapping_path = output_path / "label_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"保存标签映射到: {mapping_path}")
        
        # 创建模拟的文件清单和片段清单
        self.create_mock_catalogs(len(X), output_path)
        
        return features_path
    
    def create_mock_catalogs(self, num_samples: int, output_path: Path):
        """
        创建模拟的文件清单和片段清单
        
        Args:
            num_samples: 样本数量
            output_path: 输出路径
        """
        # 创建文件清单
        file_data = []
        segments_data = []
        
        emotions = list(self.dataset_config['emotions'].keys())
        samples_per_emotion = num_samples // len(emotions)
        
        sample_idx = 0
        for emotion in emotions:
            genres = self.dataset_config['emotions'][emotion]
            for genre in genres:
                for i in range(samples_per_emotion // len(genres)):
                    if sample_idx >= num_samples:
                        break
                    
                    # 模拟文件信息
                    filename = f"{genre}_{genre}.{i:05d}.wav"
                    file_path = f"../data/raw/{genre}/{filename}"
                    
                    file_data.append({
                        'file_path': file_path,
                        'relative_path': f"{genre}/{filename}",
                        'genre': genre,
                        'emotion': emotion,
                        'file_size': 1024000,  # 模拟文件大小
                        'filename': filename
                    })
                    
                    # 模拟片段信息
                    segments_data.append({
                        'original_file': file_path,
                        'genre': genre,
                        'emotion': emotion,
                        'segment_id': 0,
                        'start_time': 0,
                        'end_time': 3,
                        'duration': 3
                    })
                    
                    sample_idx += 1
        
        # 保存文件清单
        file_df = pd.DataFrame(file_data)
        file_catalog_path = output_path / "file_catalog.csv"
        file_df.to_csv(file_catalog_path, index=False)
        self.logger.info(f"保存文件清单到: {file_catalog_path}")
        
        # 保存片段清单
        segments_df = pd.DataFrame(segments_data)
        segments_catalog_path = output_path / "segments_catalog.csv"
        segments_df.to_csv(segments_catalog_path, index=False)
        self.logger.info(f"保存片段清单到: {segments_catalog_path}")
        
        # 创建数据集统计
        stats = {
            'total_files': int(len(file_data)),
            'genres': {k: int(v) for k, v in file_df['genre'].value_counts().items()},
            'emotions': {k: int(v) for k, v in file_df['emotion'].value_counts().items()},
            'avg_file_size': 1024000,
            'total_size': int(len(file_data) * 1024000)
        }
        
        stats_path = output_path / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"保存统计信息到: {stats_path}")

def main():
    """主函数"""
    try:
        # 创建模拟数据生成器
        generator = MockDataGenerator()
        
        # 生成模拟数据
        X, emotions, label_ids = generator.generate_mock_features(num_samples=1000)
        
        # 保存数据
        features_path = generator.save_mock_data(X, emotions, label_ids)
        
        print("\n=== 模拟数据生成完成 ===")
        print(f"特征矩阵形状: {X.shape}")
        print(f"情感类别: {np.unique(emotions)}")
        print(f"数据已保存到: {features_path}")
        
    except Exception as e:
        logging.error(f"模拟数据生成过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()