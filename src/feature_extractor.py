import numpy as np
import librosa
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import pickle

class AdvancedFeatureExtractor:
    """高级音频特征提取器，支持多种特征类型和批处理"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化特征提取器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.dataset_config = self.config['dataset']
        self.feature_config = self.config['feature_extraction']
        
        # 音频参数
        self.sample_rate = self.dataset_config['sample_rate']
        self.duration = self.dataset_config['duration']
        self.segment_duration = self.dataset_config['segment_duration']
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 特征名称映射
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """构建特征名称列表"""
        self.feature_names = []
        
        # MFCC特征名称
        if self.feature_config['mfcc']['enabled']:
            n_mfcc = self.feature_config['mfcc']['n_mfcc']
            for i in range(n_mfcc):
                for stat in ['mean', 'std', 'max', 'min']:
                    self.feature_names.append(f'mfcc_{i}_{stat}')
        
        # 频谱特征名称
        if self.feature_config['spectral']['enabled']:
            for feature in self.feature_config['spectral']['features']:
                for stat in ['mean', 'std', 'max', 'min']:
                    self.feature_names.append(f'spectral_{feature}_{stat}')
        
        # 色度特征名称
        if self.feature_config['chroma']['enabled']:
            n_chroma = self.feature_config['chroma']['n_chroma']
            for i in range(n_chroma):
                for stat in ['mean', 'std']:
                    self.feature_names.append(f'chroma_{i}_{stat}')
        
        # 节奏特征名称
        if self.feature_config['rhythm']['enabled']:
            for feature in self.feature_config['rhythm']['features']:
                if feature == 'tempo':
                    self.feature_names.append('tempo')
                elif feature == 'beat_frames':
                    self.feature_names.extend(['beat_strength_mean', 'beat_strength_std'])
                elif feature == 'onset_strength':
                    self.feature_names.extend(['onset_strength_mean', 'onset_strength_std'])
        
        # Mel频谱特征名称
        if self.feature_config['mel_spectrogram']['enabled']:
            n_mels = self.feature_config['mel_spectrogram']['n_mels']
            for i in range(min(n_mels, 20)):  # 只取前20个mel频段的统计特征
                for stat in ['mean', 'std']:
                    self.feature_names.append(f'mel_{i}_{stat}')
        
        self.logger.info(f"构建了 {len(self.feature_names)} 个特征名称")
    
    def load_audio_segment(self, file_path: str, start_time: float = 0, duration: Optional[float] = None) -> np.ndarray:
        """
        加载音频片段
        
        Args:
            file_path: 音频文件路径
            start_time: 开始时间（秒）
            duration: 持续时间（秒），None表示加载到文件结束
            
        Returns:
            音频信号
        """
        if duration is None:
            duration = self.segment_duration
        
        signal, _ = librosa.load(
            file_path,
            sr=self.sample_rate,
            offset=start_time,
            duration=duration
        )
        
        # 确保信号长度一致
        expected_length = int(self.sample_rate * duration)
        if len(signal) < expected_length:
            signal = np.pad(signal, (0, expected_length - len(signal)))
        else:
            signal = signal[:expected_length]
        
        return signal
    
    def extract_mfcc_features(self, signal: np.ndarray) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            signal: 音频信号
            
        Returns:
            MFCC特征向量
        """
        if not self.feature_config['mfcc']['enabled']:
            return np.array([])
        
        mfcc_config = self.feature_config['mfcc']
        
        mfccs = librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=mfcc_config['n_mfcc'],
            n_fft=mfcc_config['n_fft'],
            hop_length=mfcc_config['hop_length']
        )
        
        # 计算统计特征
        features = []
        features.extend(np.mean(mfccs, axis=1))  # 均值
        features.extend(np.std(mfccs, axis=1))   # 标准差
        features.extend(np.max(mfccs, axis=1))   # 最大值
        features.extend(np.min(mfccs, axis=1))   # 最小值
        
        return np.array(features)
    
    def extract_spectral_features(self, signal: np.ndarray) -> np.ndarray:
        """
        提取频谱特征
        
        Args:
            signal: 音频信号
            
        Returns:
            频谱特征向量
        """
        if not self.feature_config['spectral']['enabled']:
            return np.array([])
        
        spectral_config = self.feature_config['spectral']
        features = []
        
        # 频谱质心
        if 'centroid' in spectral_config['features']:
            centroid = librosa.feature.spectral_centroid(
                y=signal,
                sr=self.sample_rate,
                n_fft=spectral_config['n_fft'],
                hop_length=spectral_config['hop_length']
            )[0]
            features.extend([np.mean(centroid), np.std(centroid), np.max(centroid), np.min(centroid)])
        
        # 频谱带宽
        if 'bandwidth' in spectral_config['features']:
            bandwidth = librosa.feature.spectral_bandwidth(
                y=signal,
                sr=self.sample_rate,
                n_fft=spectral_config['n_fft'],
                hop_length=spectral_config['hop_length']
            )[0]
            features.extend([np.mean(bandwidth), np.std(bandwidth), np.max(bandwidth), np.min(bandwidth)])
        
        # 频谱滚降
        if 'rolloff' in spectral_config['features']:
            rolloff = librosa.feature.spectral_rolloff(
                y=signal,
                sr=self.sample_rate,
                n_fft=spectral_config['n_fft'],
                hop_length=spectral_config['hop_length']
            )[0]
            features.extend([np.mean(rolloff), np.std(rolloff), np.max(rolloff), np.min(rolloff)])
        
        # 频谱对比度
        if 'contrast' in spectral_config['features']:
            contrast = librosa.feature.spectral_contrast(
                y=signal,
                sr=self.sample_rate,
                n_fft=spectral_config['n_fft'],
                hop_length=spectral_config['hop_length']
            )
            features.extend([np.mean(contrast), np.std(contrast), np.max(contrast), np.min(contrast)])
        
        # 频谱平坦度
        if 'flatness' in spectral_config['features']:
            flatness = librosa.feature.spectral_flatness(
                y=signal,
                n_fft=spectral_config['n_fft'],
                hop_length=spectral_config['hop_length']
            )[0]
            features.extend([np.mean(flatness), np.std(flatness), np.max(flatness), np.min(flatness)])
        
        return np.array(features)
    
    def extract_chroma_features(self, signal: np.ndarray) -> np.ndarray:
        """
        提取色度特征
        
        Args:
            signal: 音频信号
            
        Returns:
            色度特征向量
        """
        if not self.feature_config['chroma']['enabled']:
            return np.array([])
        
        chroma_config = self.feature_config['chroma']
        
        chroma = librosa.feature.chroma_stft(
            y=signal,
            sr=self.sample_rate,
            n_chroma=chroma_config['n_chroma'],
            n_fft=chroma_config['n_fft'],
            hop_length=chroma_config['hop_length']
        )
        
        # 计算统计特征
        features = []
        features.extend(np.mean(chroma, axis=1))  # 均值
        features.extend(np.std(chroma, axis=1))   # 标准差
        
        return np.array(features)
    
    def extract_rhythm_features(self, signal: np.ndarray) -> np.ndarray:
        """
        提取节奏特征
        
        Args:
            signal: 音频信号
            
        Returns:
            节奏特征向量
        """
        if not self.feature_config['rhythm']['enabled']:
            return np.array([])
        
        rhythm_config = self.feature_config['rhythm']
        features = []
        
        # 节拍速度
        if 'tempo' in rhythm_config['features']:
            tempo, _ = librosa.beat.beat_track(y=signal, sr=self.sample_rate)
            features.append(tempo)
        
        # 节拍强度
        if 'beat_frames' in rhythm_config['features']:
            _, beat_frames = librosa.beat.beat_track(y=signal, sr=self.sample_rate)
            if len(beat_frames) > 1:
                beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
                beat_intervals = np.diff(beat_times)
                features.extend([np.mean(beat_intervals), np.std(beat_intervals)])
            else:
                features.extend([0.0, 0.0])
        
        # 起始强度
        if 'onset_strength' in rhythm_config['features']:
            onset_strength = librosa.onset.onset_strength(y=signal, sr=self.sample_rate)
            features.extend([np.mean(onset_strength), np.std(onset_strength)])
        
        return np.array(features)
    
    def extract_mel_spectrogram_features(self, signal: np.ndarray) -> np.ndarray:
        """
        提取Mel频谱图特征
        
        Args:
            signal: 音频信号
            
        Returns:
            Mel频谱特征向量
        """
        if not self.feature_config['mel_spectrogram']['enabled']:
            return np.array([])
        
        mel_config = self.feature_config['mel_spectrogram']
        
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_mels=mel_config['n_mels'],
            n_fft=mel_config['n_fft'],
            hop_length=mel_config['hop_length']
        )
        
        # 转换为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # 只取前20个mel频段的统计特征（减少特征维度）
        n_mels_reduced = min(mel_config['n_mels'], 20)
        features = []
        
        for i in range(n_mels_reduced):
            features.extend([
                np.mean(log_mel_spec[i]),
                np.std(log_mel_spec[i])
            ])
        
        return np.array(features)
    
    def extract_features_from_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        从音频信号提取所有特征
        
        Args:
            signal: 音频信号
            
        Returns:
            完整的特征向量
        """
        features = []
        
        # 提取各类特征
        mfcc_features = self.extract_mfcc_features(signal)
        spectral_features = self.extract_spectral_features(signal)
        chroma_features = self.extract_chroma_features(signal)
        rhythm_features = self.extract_rhythm_features(signal)
        mel_features = self.extract_mel_spectrogram_features(signal)
        
        # 合并所有特征
        for feature_set in [mfcc_features, spectral_features, chroma_features, rhythm_features, mel_features]:
            if len(feature_set) > 0:
                features.extend(feature_set)
        
        return np.array(features)
    
    def extract_features_from_file(self, file_path: str, start_time: float = 0, duration: Optional[float] = None) -> np.ndarray:
        """
        从音频文件提取特征
        
        Args:
            file_path: 音频文件路径
            start_time: 开始时间（秒）
            duration: 持续时间（秒）
            
        Returns:
            特征向量
        """
        try:
            signal = self.load_audio_segment(file_path, start_time, duration)
            return self.extract_features_from_signal(signal)
        except Exception as e:
            self.logger.error(f"提取特征时出错 {file_path}: {str(e)}")
            # 返回零向量
            return np.zeros(len(self.feature_names))
    
    def process_segment_row(self, row: pd.Series) -> Tuple[np.ndarray, str, int]:
        """
        处理单个片段行（用于并行处理）
        
        Args:
            row: 包含片段信息的pandas Series
            
        Returns:
            特征向量、情感标签、标签ID
        """
        features = self.extract_features_from_file(
            row['original_file'],
            row['start_time'],
            row['duration']
        )
        return features, row['emotion'], row.get('label_id', -1)
    
    def extract_features_batch(self, segments_df: pd.DataFrame, max_workers: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量提取特征（支持多线程）
        
        Args:
            segments_df: 包含片段信息的DataFrame
            max_workers: 最大工作线程数
            
        Returns:
            特征矩阵、情感标签数组、标签ID数组
        """
        self.logger.info(f"开始批量提取特征，共 {len(segments_df)} 个片段")
        
        features_list = []
        emotions_list = []
        label_ids_list = []
        
        # 使用线程池进行并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.process_segment_row, row): idx
                for idx, row in segments_df.iterrows()
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_idx):
                try:
                    features, emotion, label_id = future.result()
                    features_list.append(features)
                    emotions_list.append(emotion)
                    label_ids_list.append(label_id)
                    
                    completed += 1
                    if completed % 100 == 0:
                        self.logger.info(f"已完成 {completed}/{len(segments_df)} 个片段的特征提取")
                        
                except Exception as e:
                    self.logger.error(f"处理片段时出错: {str(e)}")
                    # 添加零向量
                    features_list.append(np.zeros(len(self.feature_names)))
                    emotions_list.append("unknown")
                    label_ids_list.append(-1)
        
        # 转换为numpy数组
        X = np.array(features_list)
        emotions = np.array(emotions_list)
        label_ids = np.array(label_ids_list)
        
        self.logger.info(f"特征提取完成，特征矩阵形状: {X.shape}")
        
        return X, emotions, label_ids
    
    def save_features(self, X: np.ndarray, emotions: np.ndarray, label_ids: np.ndarray, output_path: str):
        """
        保存提取的特征
        
        Args:
            X: 特征矩阵
            emotions: 情感标签
            label_ids: 标签ID
            output_path: 输出路径
        """
        # 创建特征DataFrame
        feature_df = pd.DataFrame(X, columns=self.feature_names)
        feature_df['emotion'] = emotions
        feature_df['label_id'] = label_ids
        
        # 保存为CSV
        feature_df.to_csv(output_path, index=False)
        self.logger.info(f"特征保存到: {output_path}")
        
        # 同时保存为pickle格式（更快的加载速度）
        pickle_path = output_path.replace('.csv', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'features': X,
                'emotions': emotions,
                'label_ids': label_ids,
                'feature_names': self.feature_names
            }, f)
        
        self.logger.info(f"特征二进制文件保存到: {pickle_path}")
    
    def get_feature_summary(self) -> Dict:
        """
        获取特征提取的摘要信息
        
        Returns:
            特征摘要字典
        """
        enabled_features = []
        feature_counts = {}
        
        for feature_type, config in self.feature_config.items():
            if config.get('enabled', False):
                enabled_features.append(feature_type)
                
                if feature_type == 'mfcc':
                    feature_counts[feature_type] = config['n_mfcc'] * 4  # mean, std, max, min
                elif feature_type == 'spectral':
                    feature_counts[feature_type] = len(config['features']) * 4
                elif feature_type == 'chroma':
                    feature_counts[feature_type] = config['n_chroma'] * 2  # mean, std
                elif feature_type == 'rhythm':
                    count = 0
                    if 'tempo' in config['features']:
                        count += 1
                    if 'beat_frames' in config['features']:
                        count += 2
                    if 'onset_strength' in config['features']:
                        count += 2
                    feature_counts[feature_type] = count
                elif feature_type == 'mel_spectrogram':
                    feature_counts[feature_type] = min(config['n_mels'], 20) * 2
        
        return {
            'enabled_features': enabled_features,
            'feature_counts': feature_counts,
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names
        }

def main():
    """主函数"""
    try:
        # 创建特征提取器
        extractor = AdvancedFeatureExtractor()
        
        # 显示特征摘要
        summary = extractor.get_feature_summary()
        print("\n=== 特征提取器配置 ===")
        print(f"启用的特征类型: {summary['enabled_features']}")
        print(f"各类特征数量: {summary['feature_counts']}")
        print(f"总特征数量: {summary['total_features']}")
        
        # 读取预处理后的片段数据
        segments_path = Path("data/processed/segments_catalog.csv")
        if segments_path.exists():
            segments_df = pd.read_csv(segments_path)
            
            # 提取特征
            print(f"\n开始提取 {len(segments_df)} 个片段的特征...")
            X, emotions, label_ids = extractor.extract_features_batch(segments_df)
            
            # 保存特征
            features_path = "data/processed/features.csv"
            extractor.save_features(X, emotions, label_ids, features_path)
            
            print(f"\n=== 特征提取完成 ===")
            print(f"特征矩阵形状: {X.shape}")
            print(f"情感类别: {np.unique(emotions[emotions != 'unknown'])}")
            
        else:
            print("未找到片段数据文件，请先运行数据预处理")
            
    except Exception as e:
        logging.error(f"特征提取过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()