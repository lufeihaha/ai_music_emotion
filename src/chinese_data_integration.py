#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文音乐数据集集成模块
整合PMEmo和PSIC3839数据集到训练流程中
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import librosa
from sklearn.preprocessing import LabelEncoder
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseDatasetIntegrator:
    """中文音乐数据集集成器"""
    
    def __init__(self):
        self.datasets_dir = Path("data/chinese_datasets")
        self.processed_dir = Path("data/processed")
        self.emotion_mapping = {
            # PMEmo数据集情感映射
            'happy': '快乐',
            'sad': '忧郁', 
            'angry': '激烈',
            'peaceful': '平静',
            'excited': '激昂',
            'nostalgic': '怀念',
            'romantic': '浪漫',
            'mysterious': '深沉',
            
            # PSIC3839数据集情感映射
            'joy': '快乐',
            'sadness': '忧郁',
            'anger': '激烈',
            'fear': '深沉',
            'disgust': '激烈',
            'surprise': '激昂',
            'neutral': '平静',
            'love': '浪漫',
            'longing': '怀念'
        }
        
    def check_datasets_availability(self) -> Dict[str, bool]:
        """检查中文数据集的可用性"""
        availability = {}
        
        # 检查PMEmo数据集
        pmemo_path = self.datasets_dir / "PMEmo"
        availability['PMEmo'] = pmemo_path.exists()
        
        # 检查PSIC3839数据集
        psic_path = self.datasets_dir / "PSIC3839"
        availability['PSIC3839'] = psic_path.exists()
        
        logger.info(f"数据集可用性检查: {availability}")
        return availability
    
    def load_pmemo_dataset(self) -> Optional[pd.DataFrame]:
        """加载PMEmo数据集"""
        try:
            pmemo_path = self.datasets_dir / "PMEmo"
            if not pmemo_path.exists():
                logger.warning("PMEmo数据集不存在")
                return None
            
            # 查找可能的标注文件
            annotation_files = list(pmemo_path.glob("*.csv")) + list(pmemo_path.glob("*.xlsx"))
            
            if not annotation_files:
                logger.warning("PMEmo数据集中未找到标注文件")
                return None
            
            # 尝试加载第一个找到的标注文件
            for file_path in annotation_files:
                try:
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    
                    logger.info(f"成功加载PMEmo数据集: {file_path}")
                    logger.info(f"数据集形状: {df.shape}")
                    logger.info(f"列名: {df.columns.tolist()}")
                    return df
                    
                except Exception as e:
                    logger.warning(f"加载 {file_path} 失败: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"加载PMEmo数据集时发生错误: {str(e)}")
            return None
    
    def load_psic3839_dataset(self) -> Optional[pd.DataFrame]:
        """加载PSIC3839数据集"""
        try:
            psic_path = self.datasets_dir / "PSIC3839"
            if not psic_path.exists():
                logger.warning("PSIC3839数据集不存在")
                return None
            
            # 查找标注文件
            annotation_files = list(psic_path.glob("*.xlsx")) + list(psic_path.glob("*.csv"))
            
            if not annotation_files:
                logger.warning("PSIC3839数据集中未找到标注文件")
                return None
            
            all_data = []
            
            for file_path in annotation_files:
                try:
                    if file_path.suffix == '.xlsx':
                        df = pd.read_excel(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    
                    # 添加文件来源信息
                    df['source_file'] = file_path.name
                    all_data.append(df)
                    
                    logger.info(f"加载PSIC3839文件: {file_path}")
                    logger.info(f"数据形状: {df.shape}")
                    
                except Exception as e:
                    logger.warning(f"加载 {file_path} 失败: {str(e)}")
                    continue
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"PSIC3839数据集合并完成，总数据量: {combined_df.shape}")
                logger.info(f"列名: {combined_df.columns.tolist()}")
                return combined_df
            
            return None
            
        except Exception as e:
            logger.error(f"加载PSIC3839数据集时发生错误: {str(e)}")
            return None
    
    def analyze_dataset_structure(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """分析数据集结构"""
        analysis = {
            'dataset_name': dataset_name,
            'total_samples': len(df),
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head().to_dict()
        }
        
        # 尝试识别情感相关的列
        emotion_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['emotion', 'feeling', 'mood', 'valence', 'arousal']):
                emotion_columns.append(col)
        
        analysis['emotion_columns'] = emotion_columns
        
        # 如果有情感列，分析情感分布
        if emotion_columns:
            for col in emotion_columns:
                if df[col].dtype == 'object':
                    analysis[f'{col}_distribution'] = df[col].value_counts().to_dict()
                else:
                    analysis[f'{col}_stats'] = df[col].describe().to_dict()
        
        return analysis
    
    def create_unified_emotion_labels(self, pmemo_df: Optional[pd.DataFrame], 
                                    psic_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """创建统一的情感标签数据"""
        unified_data = []
        
        # 处理PMEmo数据
        if pmemo_df is not None:
            logger.info("处理PMEmo数据...")
            # 这里需要根据实际的PMEmo数据结构进行调整
            # 假设有文件路径和情感标签列
            for _, row in pmemo_df.iterrows():
                try:
                    # 根据实际数据结构调整
                    file_path = row.get('file_path', row.get('filename', ''))
                    emotion = row.get('emotion', row.get('label', ''))
                    
                    if file_path and emotion:
                        # 映射到统一的中文情感标签
                        unified_emotion = self.emotion_mapping.get(emotion, emotion)
                        
                        unified_data.append({
                            'file_path': file_path,
                            'emotion': unified_emotion,
                            'source': 'PMEmo',
                            'original_emotion': emotion,
                            'confidence': 1.0  # PMEmo数据假设为高置信度
                        })
                except Exception as e:
                    logger.warning(f"处理PMEmo数据行时出错: {str(e)}")
                    continue
        
        # 处理PSIC3839数据
        if psic_df is not None:
            logger.info("处理PSIC3839数据...")
            for _, row in psic_df.iterrows():
                try:
                    # 根据实际数据结构调整
                    file_path = row.get('file_path', row.get('filename', ''))
                    emotion = row.get('emotion', row.get('label', ''))
                    
                    if file_path and emotion:
                        # 映射到统一的中文情感标签
                        unified_emotion = self.emotion_mapping.get(emotion, emotion)
                        
                        unified_data.append({
                            'file_path': file_path,
                            'emotion': unified_emotion,
                            'source': 'PSIC3839',
                            'original_emotion': emotion,
                            'confidence': 1.0
                        })
                except Exception as e:
                    logger.warning(f"处理PSIC3839数据行时出错: {str(e)}")
                    continue
        
        if unified_data:
            unified_df = pd.DataFrame(unified_data)
            logger.info(f"创建统一数据集完成，总样本数: {len(unified_df)}")
            logger.info(f"情感分布: {unified_df['emotion'].value_counts().to_dict()}")
            return unified_df
        else:
            logger.warning("没有可用的数据创建统一数据集")
            return pd.DataFrame()
    
    def extract_features_from_chinese_data(self, unified_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """从中文数据中提取特征"""
        features_list = []
        labels_list = []
        
        logger.info("开始从中文数据提取特征...")
        
        for idx, row in unified_df.iterrows():
            try:
                file_path = row['file_path']
                emotion = row['emotion']
                
                # 检查文件是否存在
                full_path = self.datasets_dir / file_path
                if not full_path.exists():
                    # 尝试其他可能的路径
                    alt_paths = [
                        self.datasets_dir / "PMEmo" / file_path,
                        self.datasets_dir / "PSIC3839" / file_path,
                        Path(file_path)
                    ]
                    
                    found = False
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            full_path = alt_path
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"文件不存在: {file_path}")
                        continue
                
                # 提取音频特征
                features = self.extract_audio_features(str(full_path))
                if features is not None:
                    features_list.append(features)
                    labels_list.append(emotion)
                    
                    if len(features_list) % 10 == 0:
                        logger.info(f"已处理 {len(features_list)} 个文件")
                
            except Exception as e:
                logger.warning(f"处理文件 {row['file_path']} 时出错: {str(e)}")
                continue
        
        if features_list:
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)
            
            logger.info(f"特征提取完成，特征形状: {features_array.shape}")
            logger.info(f"标签数量: {len(labels_array)}")
            
            return features_array, labels_array
        else:
            logger.warning("没有成功提取任何特征")
            return np.array([]), np.array([])
    
    def extract_audio_features(self, audio_path: str) -> Optional[np.ndarray]:
        """提取单个音频文件的特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            
            # 提取基本特征
            features = []
            
            # MFCC特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff)
            ])
            
            # 色度特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            
            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # RMS能量
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"提取音频特征失败 {audio_path}: {str(e)}")
            return None
    
    def save_integrated_dataset(self, features: np.ndarray, labels: np.ndarray, 
                              metadata: Dict) -> None:
        """保存集成后的数据集"""
        try:
            # 创建输出目录
            output_dir = Path("data/chinese_integrated")
            output_dir.mkdir(exist_ok=True)
            
            # 保存特征和标签
            np.save(output_dir / "features.npy", features)
            np.save(output_dir / "labels.npy", labels)
            
            # 保存元数据
            with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 创建标签编码器
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            
            # 保存标签编码器
            with open(output_dir / "label_encoder.pkl", 'wb') as f:
                pickle.dump(label_encoder, f)
            
            # 保存编码后的标签
            np.save(output_dir / "encoded_labels.npy", encoded_labels)
            
            logger.info(f"集成数据集已保存到: {output_dir}")
            logger.info(f"特征形状: {features.shape}")
            logger.info(f"标签数量: {len(labels)}")
            logger.info(f"唯一情感: {np.unique(labels)}")
            
        except Exception as e:
            logger.error(f"保存集成数据集时出错: {str(e)}")
    
    def generate_integration_report(self, availability: Dict, pmemo_df: Optional[pd.DataFrame],
                                  psic_df: Optional[pd.DataFrame], unified_df: pd.DataFrame,
                                  features: np.ndarray, labels: np.ndarray) -> None:
        """生成集成报告"""
        try:
            report = {
                'integration_timestamp': pd.Timestamp.now().isoformat(),
                'dataset_availability': availability,
                'data_statistics': {
                    'total_samples': len(unified_df) if not unified_df.empty else 0,
                    'feature_dimensions': features.shape if features.size > 0 else None,
                    'emotion_distribution': unified_df['emotion'].value_counts().to_dict() if not unified_df.empty else {}
                }
            }
            
            # 分析PMEmo数据
            if pmemo_df is not None:
                report['pmemo_analysis'] = self.analyze_dataset_structure(pmemo_df, 'PMEmo')
            
            # 分析PSIC3839数据
            if psic_df is not None:
                report['psic_analysis'] = self.analyze_dataset_structure(psic_df, 'PSIC3839')
            
            # 保存报告
            report_path = Path("data/chinese_integrated/integration_report.json")
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"集成报告已保存到: {report_path}")
            
        except Exception as e:
            logger.error(f"生成集成报告时出错: {str(e)}")

def main():
    """主函数"""
    print("🇨🇳 中文音乐数据集集成工具")
    print("=" * 50)
    
    integrator = ChineseDatasetIntegrator()
    
    # 检查数据集可用性
    availability = integrator.check_datasets_availability()
    
    if not any(availability.values()):
        print("❌ 没有找到可用的中文数据集")
        print("请确保以下数据集存在：")
        print("- data/chinese_datasets/PMEmo/")
        print("- data/chinese_datasets/PSIC3839/")
        return
    
    # 加载数据集
    pmemo_df = integrator.load_pmemo_dataset() if availability['PMEmo'] else None
    psic_df = integrator.load_psic3839_dataset() if availability['PSIC3839'] else None
    
    if pmemo_df is None and psic_df is None:
        print("❌ 无法加载任何数据集")
        return
    
    # 创建统一标签
    unified_df = integrator.create_unified_emotion_labels(pmemo_df, psic_df)
    
    if unified_df.empty:
        print("❌ 无法创建统一的情感标签")
        return
    
    # 提取特征
    features, labels = integrator.extract_features_from_chinese_data(unified_df)
    
    if features.size == 0:
        print("❌ 特征提取失败")
        return
    
    # 保存集成数据
    metadata = {
        'dataset_sources': [k for k, v in availability.items() if v],
        'total_samples': len(unified_df),
        'feature_dimensions': features.shape[1] if len(features.shape) > 1 else features.shape[0],
        'emotion_labels': np.unique(labels).tolist()
    }
    
    integrator.save_integrated_dataset(features, labels, metadata)
    
    # 生成报告
    integrator.generate_integration_report(availability, pmemo_df, psic_df, unified_df, features, labels)
    
    print("\n✅ 中文数据集集成完成！")
    print(f"📊 总样本数: {len(unified_df)}")
    print(f"🎵 特征维度: {features.shape}")
    print(f"😊 情感类别: {np.unique(labels)}")
    print("📁 结果保存在 data/chinese_integrated/ 目录")

if __name__ == "__main__":
    main() 