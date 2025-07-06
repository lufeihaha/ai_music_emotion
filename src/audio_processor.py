import numpy as np
import librosa
from typing import Tuple, List, Dict
import os
import logging

class AudioProcessor:
    def __init__(self, sr: int = 22050, duration: int = 30, hop_length: int = 512):
        """
        初始化音频处理器
        
        Args:
            sr: 采样率
            duration: 音频片段长度（秒）
            hop_length: 帧长
        """
        self.sr = sr
        self.duration = duration
        self.hop_length = hop_length
        self.frame_length = 2048
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频信号
        """
        signal, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
        
        # 确保所有音频长度一致
        if len(signal) < self.frame_length:
            signal = np.pad(signal, (0, self.frame_length - len(signal)))
        else:
            signal = signal[:self.frame_length]
            
        return signal
    
    def process_audio_file(self, file_path: str) -> np.ndarray:
        """
        处理单个音频文件并提取特征
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            特征向量
        """
        try:
            # 加载音频
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            # 1. 基础特征
            mfcc = self.extract_mfcc(y)
            spectral = self.extract_spectral_features(y)
            rhythm = self.extract_rhythm_features(y)
            harmony = self.extract_harmony_features(y)
            
            # 2. 情感相关特征
            emotion_features = self.extract_emotion_features(y)
            
            # 3. 时序变化特征
            temporal_features = self.extract_temporal_features(y)
            
            # 合并所有特征
            all_features = np.concatenate([
                mfcc,
                spectral,
                rhythm,
                harmony,
                emotion_features,
                temporal_features
            ])
            
            return all_features
            
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {str(e)}")
            raise
    
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """提取MFCC特征"""
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 计算统计特征
        mfcc_features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.std(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1),
            np.std(mfcc_delta2, axis=1)
        ])
        
        return mfcc_features
    
    def extract_spectral_features(self, y: np.ndarray) -> np.ndarray:
        """提取频谱特征"""
        # 频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        # 频谱衰减
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        # 频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        # 频谱对比度
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        
        # 计算统计特征
        spectral_features = np.concatenate([
            [np.mean(spectral_centroids), np.std(spectral_centroids)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            np.mean(spectral_contrast, axis=1),
            np.std(spectral_contrast, axis=1)
        ])
        
        return spectral_features
    
    def extract_rhythm_features(self, y: np.ndarray) -> np.ndarray:
        """提取节奏特征"""
        # 节奏特征
        tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
        
        # 计算onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=self.sr)
        
        # 计算节奏规律性
        rhythm_regularity = np.std(np.diff(pulse))
        
        # 计算节奏强度变化
        onset_strength = np.mean(onset_env)
        onset_strength_std = np.std(onset_env)
        
        rhythm_features = np.array([
            tempo,
            rhythm_regularity,
            onset_strength,
            onset_strength_std
        ])
        
        return rhythm_features
    
    def extract_harmony_features(self, y: np.ndarray) -> np.ndarray:
        """提取和声特征"""
        # 色度特征
        chromagram = librosa.feature.chroma_stft(y=y, sr=self.sr)
        
        # 调性强度
        tonnetz = librosa.feature.tonnetz(y=y, sr=self.sr)
        
        # 和声复杂度
        harmony_complexity = np.std(chromagram, axis=1)
        
        # 计算统计特征
        harmony_features = np.concatenate([
            np.mean(chromagram, axis=1),
            np.std(chromagram, axis=1),
            np.mean(tonnetz, axis=1),
            np.std(tonnetz, axis=1),
            harmony_complexity
        ])
        
        return harmony_features
    
    def extract_emotion_features(self, y: np.ndarray) -> np.ndarray:
        """提取情感相关特征"""
        # 1. 能量包络 - 与情感强度相关
        rms = librosa.feature.rms(y=y)[0]
        
        # 2. 音高变化 - 与情感起伏相关
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                        fmax=librosa.note_to_hz('C7'),
                                        sr=self.sr)
        f0 = f0[voiced_flag]
        
        # 3. 谐波噪声比 - 与音色纯度相关
        hnr = np.mean(librosa.effects.harmonic(y))
        
        # 4. 响度动态范围 - 与情感表达强度相关
        percentile_20 = np.percentile(rms, 20)
        percentile_80 = np.percentile(rms, 80)
        dynamic_range = percentile_80 - percentile_20
        
        # 5. 音高变化率 - 与情感变化速度相关
        if len(f0) > 0:
            pitch_changes = np.diff(f0)
            pitch_change_rate = np.mean(np.abs(pitch_changes))
        else:
            pitch_change_rate = 0
            
        emotion_features = np.array([
            np.mean(rms),  # 平均能量
            np.std(rms),   # 能量变化
            np.mean(f0) if len(f0) > 0 else 0,  # 平均音高
            np.std(f0) if len(f0) > 0 else 0,   # 音高变化
            hnr,          # 谐波噪声比
            dynamic_range,  # 响度动态范围
            pitch_change_rate  # 音高变化率
        ])
        
        return emotion_features
    
    def extract_temporal_features(self, y: np.ndarray) -> np.ndarray:
        """提取时序变化特征"""
        # 将信号分成多个时间窗口
        frame_length = int(self.sr * 3)  # 3秒窗口
        hop_length = int(frame_length / 2)  # 50% 重叠
        
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        
        # 对每个窗口计算特征
        frame_features = []
        for frame in frames.T:
            # 1. 能量变化
            rms = librosa.feature.rms(y=frame)[0]
            
            # 2. 频谱变化
            spec_cent = librosa.feature.spectral_centroid(y=frame, sr=self.sr)[0]
            
            # 3. 音高变化
            f0, voiced_flag, _ = librosa.pyin(frame, 
                                            fmin=librosa.note_to_hz('C2'),
                                            fmax=librosa.note_to_hz('C7'),
                                            sr=self.sr)
            f0 = f0[voiced_flag]
            
            frame_features.append([
                np.mean(rms),
                np.std(rms),
                np.mean(spec_cent),
                np.std(spec_cent),
                np.mean(f0) if len(f0) > 0 else 0,
                np.std(f0) if len(f0) > 0 else 0
            ])
        
        # 计算时序特征的统计量
        frame_features = np.array(frame_features)
        temporal_features = np.concatenate([
            np.mean(frame_features, axis=0),
            np.std(frame_features, axis=0),
            np.max(frame_features, axis=0) - np.min(frame_features, axis=0)  # 动态范围
        ])
        
        return temporal_features
    
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
                
                features.append(feature_vector)
                labels.append(genre_id)
        
        return np.array(features), np.array(labels)