import numpy as np
import librosa
from typing import Tuple, List, Dict, Any, Optional
import os

class AudioProcessor:
    def __init__(self, sr: int = 22050, duration: int = 30):
        """
        初始化音频处理器
        
        Args:
            sr: 采样率
            duration: 音频片段长度（秒）
        """
        self.sr = sr
        self.duration = duration
        self.n_samples = sr * duration
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频信号
        """
        try:
            signal, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            # 确保所有音频长度一致
            if len(signal) < self.n_samples:
                signal = np.pad(signal, (0, self.n_samples - len(signal)))
            else:
                signal = signal[:self.n_samples]
                
            return signal
        except Exception as e:
            print(f"Error loading audio file {file_path}: {str(e)}")
            return None
    
    def extract_features(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        提取音频特征
        
        Args:
            signal: 音频信号
            
        Returns:
            MFCC特征, 频谱质心, 色度特征, 过零率
        """
        try:
            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(y=signal, sr=self.sr, n_mfcc=13)
            
            # 提取频谱质心
            spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=self.sr)[0]
            
            # 提取色度特征
            chromagram = librosa.feature.chroma_stft(y=signal, sr=self.sr)
            
            # 计算过零率
            zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)[0]
            
            return mfccs, spectral_centroids, chromagram, zero_crossing_rate
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None, None, None, None
    
    def extract_emotion_features(self, signal: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        提取情感分析所需的特征
        
        Args:
            signal: 音频信号
            
        Returns:
            特征字典，包含MFCC、频谱质心、色度特征等
        """
        try:
            # 提取基础特征
            mfccs, spec_centroids, chroma, zcr = self.extract_features(signal)
            
            if mfccs is None:
                return None
            
            # 计算RMS能量
            rmse = librosa.feature.rms(y=signal)[0]
            
            # 返回特征字典格式，符合情感分类器的期望
            features = {
                'mfcc': np.mean(mfccs, axis=1),  # 取MFCC的均值
                'spectral_centroid': np.mean(spec_centroids),
                'chroma': np.mean(chroma, axis=1),  # 取色度特征的均值
                'zero_crossing_rate': np.mean(zcr),
                'rmse': np.mean(rmse)
            }
            
            return features
        except Exception as e:
            print(f"Error extracting emotion features: {str(e)}")
            return None
    
    def get_emotion_features(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        从音频文件中提取情感特征序列（每3秒一个特征）
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            特征序列列表
        """
        try:
            # 加载完整音频
            signal, _ = librosa.load(file_path, sr=self.sr)
            
            # 分段提取特征（每3秒一段）
            segment_duration = 3  # 秒
            segment_samples = segment_duration * self.sr
            features_sequence = []
            
            # 确保至少有一段
            if len(signal) < segment_samples:
                signal = np.pad(signal, (0, segment_samples - len(signal)))
            
            # 按3秒分段
            for i in range(0, len(signal), segment_samples):
                segment = signal[i:i + segment_samples]
                
                # 如果最后一段不足3秒，补零
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)))
                
                # 提取该段的特征
                features = self.extract_emotion_features(segment)
                if features is not None:
                    features_sequence.append(features)
            
            return features_sequence if features_sequence else None
            
        except Exception as e:
            print(f"Error processing audio file {file_path}: {str(e)}")
            return None

    def process_audio_file(self, file_path: str) -> np.ndarray:
        """
        处理单个音频文件并提取特征
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            特征向量
        """
        # 加载音频
        signal = self.load_audio(file_path)
        if signal is None:
            return None
        
        # 提取特征
        mfccs, spec_centroids, chroma, zcr = self.extract_features(signal)
        if mfccs is None:
            return None
        
        # 计算统计特征
        features = []
        
        # MFCC统计特征
        features.extend([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.max(mfccs, axis=1),
            np.min(mfccs, axis=1)
        ])
        
        # 频谱质心统计特征
        features.extend([
            np.mean(spec_centroids),
            np.std(spec_centroids),
            np.max(spec_centroids),
            np.min(spec_centroids)
        ])
        
        # 色度特征统计特征
        features.extend([
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        
        # 过零率统计特征
        features.extend([
            np.mean(zcr),
            np.std(zcr),
            np.max(zcr),
            np.min(zcr)
        ])
        
        return np.concatenate([f.flatten() for f in features])
    
    def process_dataset(self, audio_dir: str, genres: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理整个数据集
        
        Args:
            audio_dir: 音频文件目录
            genres: 音乐流派列表
            
        Returns:
            特征矩阵和标签数组
        """
        features = []
        labels = []
        
        for genre_id, genre in enumerate(genres):
            genre_path = os.path.join(audio_dir, genre)
            if not os.path.exists(genre_path):
                continue
                
            for file_name in os.listdir(genre_path):
                if not file_name.endswith('.wav'):
                    continue
                    
                file_path = os.path.join(genre_path, file_name)
                feature_vector = self.process_audio_file(file_path)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(genre_id)
        
        return np.array(features), np.array(labels)