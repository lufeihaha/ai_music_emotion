import os
import json
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

class DataPreprocessor:
    """数据预处理器，负责数据加载、清理、分割和标签处理"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化数据预处理器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.dataset_config = self.config['dataset']
        self.training_config = self.config['training']
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建标签编码器
        self.label_encoder = LabelEncoder()
        self.emotion_to_id = {}
        self.id_to_emotion = {}
    
    def create_emotion_mapping(self) -> Dict:
        """
        根据流派创建情感映射
        
        Returns:
            情感映射字典
        """
        emotion_mapping = {}
        
        for emotion, genres in self.dataset_config['emotions'].items():
            for genre in genres:
                emotion_mapping[genre] = emotion
        
        self.logger.info(f"创建情感映射: {emotion_mapping}")
        return emotion_mapping
    
    def scan_dataset(self) -> pd.DataFrame:
        """
        扫描数据集并创建文件清单
        
        Returns:
            包含文件信息的DataFrame
        """
        # 如果当前在src目录，需要调整路径
        raw_path = Path(self.dataset_config['path'])
        if not raw_path.exists() and Path("../data/raw").exists():
            raw_path = Path("../data/raw")
        
        processed_path = Path(self.dataset_config['processed_path'])
        if not processed_path.parent.exists() and Path("../data").exists():
            processed_path = Path("../data/processed")
        
        # 创建处理后的数据目录
        processed_path.mkdir(parents=True, exist_ok=True)
        
        files_info = []
        emotion_mapping = self.create_emotion_mapping()
        
        # 扫描所有音频文件
        for genre in self.dataset_config['genres']:
            genre_path = raw_path / genre
            if not genre_path.exists():
                self.logger.warning(f"流派目录不存在: {genre_path}")
                continue
            
            for audio_file in genre_path.glob("*.wav"):
                emotion = emotion_mapping.get(genre, "unknown")
                
                files_info.append({
                    'file_path': str(audio_file),
                    'relative_path': str(audio_file.relative_to(raw_path)),
                    'genre': genre,
                    'emotion': emotion,
                    'file_size': audio_file.stat().st_size,
                    'filename': audio_file.name
                })
        
        df = pd.DataFrame(files_info)
        
        # 保存文件清单
        catalog_path = processed_path / "file_catalog.csv"
        df.to_csv(catalog_path, index=False)
        self.logger.info(f"保存文件清单到: {catalog_path}")
        self.logger.info(f"总计扫描到 {len(df)} 个音频文件")
        
        return df
    
    def validate_audio_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        验证音频文件的有效性
        
        Args:
            df: 文件清单DataFrame
            
        Returns:
            验证后的DataFrame
        """
        valid_files = []
        
        for idx, row in df.iterrows():
            try:
                # 尝试加载音频文件
                y, sr = librosa.load(row['file_path'], duration=1)
                if len(y) > 0:
                    valid_files.append(row)
                else:
                    self.logger.warning(f"音频文件为空: {row['file_path']}")
            except Exception as e:
                self.logger.error(f"无法加载音频文件 {row['file_path']}: {str(e)}")
        
        valid_df = pd.DataFrame(valid_files)
        self.logger.info(f"验证完成，有效文件数: {len(valid_df)}")
        
        return valid_df
    
    def create_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将长音频文件分割成短片段
        
        Args:
            df: 文件清单DataFrame
            
        Returns:
            包含片段信息的DataFrame
        """
        segments_info = []
        segment_duration = self.dataset_config['segment_duration']
        hop_length = self.dataset_config['hop_length']
        total_duration = self.dataset_config['duration']
        
        for idx, row in df.iterrows():
            # 计算可以分割的片段数
            num_segments = int((total_duration - segment_duration) / hop_length) + 1
            
            for segment_idx in range(num_segments):
                start_time = segment_idx * hop_length
                end_time = start_time + segment_duration
                
                segments_info.append({
                    'original_file': row['file_path'],
                    'genre': row['genre'],
                    'emotion': row['emotion'],
                    'segment_id': segment_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': segment_duration
                })
        
        segments_df = pd.DataFrame(segments_info)
        
        # 保存片段信息
        processed_path = Path(self.dataset_config['processed_path'])
        segments_path = processed_path / "segments_catalog.csv"
        segments_df.to_csv(segments_path, index=False)
        self.logger.info(f"保存片段清单到: {segments_path}")
        self.logger.info(f"总计创建 {len(segments_df)} 个音频片段")
        
        return segments_df
    
    def prepare_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        准备标签编码
        
        Args:
            df: 包含标签的DataFrame
            
        Returns:
            编码后的标签数组和映射字典
        """
        # 获取所有情感标签
        emotions = df['emotion'].unique()
        emotions = [e for e in emotions if e != "unknown"]
        
        # 创建标签编码
        self.label_encoder.fit(emotions)
        
        # 创建映射字典
        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotions)}
        self.id_to_emotion = {idx: emotion for idx, emotion in enumerate(emotions)}
        
        # 编码标签
        valid_mask = df['emotion'] != "unknown"
        encoded_labels = np.full(len(df), -1)
        encoded_labels[valid_mask] = self.label_encoder.transform(df.loc[valid_mask, 'emotion'])
        
        # 保存标签映射
        processed_path = Path(self.dataset_config['processed_path'])
        mapping_path = processed_path / "label_mapping.json"
        
        mapping_data = {
            'emotion_to_id': self.emotion_to_id,
            'id_to_emotion': self.id_to_emotion,
            'emotions': emotions.tolist()
        }
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"保存标签映射到: {mapping_path}")
        self.logger.info(f"情感类别: {emotions}")
        
        return encoded_labels, mapping_data
    
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            df: 数据集DataFrame
            
        Returns:
            训练集、验证集、测试集的DataFrame
        """
        # 过滤掉未知情感的样本
        valid_df = df[df['emotion'] != "unknown"].copy()
        
        # 分层分割
        if self.training_config['stratify']:
            stratify = valid_df['emotion']
        else:
            stratify = None
        
        # 首先分割出测试集
        train_val_df, test_df = train_test_split(
            valid_df,
            test_size=self.training_config['test_size'],
            random_state=self.training_config['random_state'],
            stratify=stratify
        )
        
        # 再分割训练集和验证集
        if self.training_config['stratify']:
            stratify_train_val = train_val_df['emotion']
        else:
            stratify_train_val = None
            
        val_size = self.training_config['validation_size'] / (1 - self.training_config['test_size'])
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=self.training_config['random_state'],
            stratify=stratify_train_val
        )
        
        # 保存分割结果
        processed_path = Path(self.dataset_config['processed_path'])
        train_df.to_csv(processed_path / "train_set.csv", index=False)
        val_df.to_csv(processed_path / "val_set.csv", index=False)
        test_df.to_csv(processed_path / "test_set.csv", index=False)
        
        self.logger.info(f"数据集分割完成:")
        self.logger.info(f"  训练集: {len(train_df)} 样本")
        self.logger.info(f"  验证集: {len(val_df)} 样本") 
        self.logger.info(f"  测试集: {len(test_df)} 样本")
        
        return train_df, val_df, test_df
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        生成数据集统计信息
        
        Args:
            df: 数据集DataFrame
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_files': len(df),
            'genres': df['genre'].value_counts().to_dict(),
            'emotions': df['emotion'].value_counts().to_dict(),
            'avg_file_size': df['file_size'].mean(),
            'total_size': df['file_size'].sum()
        }
        
        # 保存统计信息
        processed_path = Path(self.dataset_config['processed_path'])
        stats_path = processed_path / "dataset_statistics.json"
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"保存统计信息到: {stats_path}")
        return stats
    
    def run_preprocessing(self) -> Dict:
        """
        运行完整的数据预处理流程
        
        Returns:
            预处理结果摘要
        """
        self.logger.info("开始数据预处理...")
        
        # 1. 扫描数据集
        self.logger.info("步骤 1: 扫描数据集")
        df = self.scan_dataset()
        
        # 2. 验证音频文件
        self.logger.info("步骤 2: 验证音频文件")
        df = self.validate_audio_files(df)
        
        # 3. 创建音频片段
        self.logger.info("步骤 3: 创建音频片段")
        segments_df = self.create_segments(df)
        
        # 4. 准备标签
        self.logger.info("步骤 4: 准备标签编码")
        encoded_labels, label_mapping = self.prepare_labels(segments_df)
        segments_df['label_id'] = encoded_labels
        
        # 5. 分割数据集
        self.logger.info("步骤 5: 分割数据集")
        train_df, val_df, test_df = self.split_dataset(segments_df)
        
        # 6. 生成统计信息
        self.logger.info("步骤 6: 生成统计信息")
        stats = self.generate_statistics(df)
        
        # 保存预处理器状态
        processed_path = Path(self.dataset_config['processed_path'])
        preprocessor_path = processed_path / "preprocessor.pkl"
        
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self, f)
        
        result = {
            'total_files': len(df),
            'total_segments': len(segments_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'emotions': list(self.emotion_to_id.keys()),
            'statistics': stats
        }
        
        self.logger.info("数据预处理完成!")
        return result

def main():
    """主函数"""
    try:
        # 创建预处理器
        preprocessor = DataPreprocessor()
        
        # 运行预处理
        result = preprocessor.run_preprocessing()
        
        print("\n=== 数据预处理完成 ===")
        print(f"总文件数: {result['total_files']}")
        print(f"总片段数: {result['total_segments']}")
        print(f"训练样本: {result['train_samples']}")
        print(f"验证样本: {result['val_samples']}")
        print(f"测试样本: {result['test_samples']}")
        print(f"情感类别: {result['emotions']}")
        
    except Exception as e:
        logging.error(f"预处理过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()