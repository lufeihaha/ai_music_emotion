import numpy as np
import librosa
import random
from typing import Optional, List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class DataAugmentor:
    """音频数据增强器"""
    
    def __init__(self, sr: int = 22050):
        """
        初始化数据增强器
        
        Args:
            sr: 采样率
        """
        self.sr = sr
        
    def add_noise(self, signal: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            signal: 音频信号
            noise_factor: 噪声强度
            
        Returns:
            增强后的信号
        """
        noise = np.random.normal(0, 1, len(signal))
        augmented_signal = signal + noise_factor * noise
        return augmented_signal
    
    def time_stretch(self, signal: np.ndarray, rate: float) -> np.ndarray:
        """
        时间拉伸
        
        Args:
            signal: 音频信号
            rate: 拉伸率 (>1加快，<1减慢)
            
        Returns:
            拉伸后的信号
        """
        return librosa.effects.time_stretch(signal, rate=rate)
    
    def pitch_shift(self, signal: np.ndarray, steps: int) -> np.ndarray:
        """
        音高偏移
        
        Args:
            signal: 音频信号
            steps: 偏移步数（半音数量，可以是负数）
            
        Returns:
            偏移后的信号
        """
        return librosa.effects.pitch_shift(signal, sr=self.sr, n_steps=steps)
    
    def random_gain(self, signal: np.ndarray, min_factor: float = 0.8, max_factor: float = 1.2) -> np.ndarray:
        """
        随机增益
        
        Args:
            signal: 音频信号
            min_factor: 最小增益系数
            max_factor: 最大增益系数
            
        Returns:
            增益后的信号
        """
        gain_factor = random.uniform(min_factor, max_factor)
        return signal * gain_factor
    
    def add_background(self, signal: np.ndarray, background: np.ndarray, ratio: float = 0.1) -> np.ndarray:
        """
        添加背景音
        
        Args:
            signal: 主音频信号
            background: 背景音频信号
            ratio: 背景音强度比例
            
        Returns:
            混合后的信号
        """
        # 确保背景音长度匹配
        if len(background) > len(signal):
            background = background[:len(signal)]
        else:
            background = np.pad(background, (0, len(signal) - len(background)), 'wrap')
        
        return signal + ratio * background
    
    def augment_signal(self, signal: np.ndarray, augmentation_types: Optional[List[str]] = None) -> np.ndarray:
        """
        对信号进行随机增强
        
        Args:
            signal: 音频信号
            augmentation_types: 增强类型列表，可选['noise', 'stretch', 'pitch', 'gain']
            
        Returns:
            增强后的信号
        """
        if augmentation_types is None:
            augmentation_types = ['noise', 'stretch', 'pitch', 'gain']
        
        augmented = signal.copy()
        
        for aug_type in augmentation_types:
            if aug_type == 'noise' and random.random() < 0.5:
                augmented = self.add_noise(augmented, noise_factor=random.uniform(0.001, 0.01))
            elif aug_type == 'stretch' and random.random() < 0.5:
                augmented = self.time_stretch(augmented, rate=random.uniform(0.8, 1.2))
            elif aug_type == 'pitch' and random.random() < 0.5:
                augmented = self.pitch_shift(augmented, steps=random.randint(-4, 4))
            elif aug_type == 'gain' and random.random() < 0.5:
                augmented = self.random_gain(augmented)
        
        return augmented
    
    def generate_augmented_batch(self, signals: List[np.ndarray], 
                               augmentations_per_signal: int = 2) -> List[np.ndarray]:
        """
        为一批信号生成增强数据
        
        Args:
            signals: 原始信号列表
            augmentations_per_signal: 每个信号生成的增强数量
            
        Returns:
            增强后的信号列表
        """
        augmented_signals = []
        
        for signal in signals:
            # 添加原始信号
            augmented_signals.append(signal)
            
            # 生成增强信号
            for _ in range(augmentations_per_signal):
                aug_signal = self.augment_signal(signal)
                augmented_signals.append(aug_signal)
        
        return augmented_signals
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, 
                       augmentations_per_class: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        平衡数据集并进行数据增强
        
        Args:
            X: 特征矩阵
            y: 标签数组
            augmentations_per_class: 每个类别需要的增强数量
            
        Returns:
            增强后的特征矩阵和标签数组
        """
        X_augmented = []
        y_augmented = []
        
        # 对每个类别进行增强
        unique_labels = np.unique(y)
        for label in unique_labels:
            # 获取当前类别的样本
            mask = y == label
            X_class = X[mask]
            y_class = y[mask]
            
            # 添加原始样本
            X_augmented.extend(X_class)
            y_augmented.extend(y_class)
            
            # 进行数据增强
            n_augmentations = augmentations_per_class.get(label, 0)
            if n_augmentations > 0:
                for _ in range(n_augmentations):
                    idx = random.randint(0, len(X_class) - 1)
                    aug_signal = self.augment_signal(X_class[idx])
                    X_augmented.append(aug_signal)
                    y_augmented.append(label)
        
        return np.array(X_augmented), np.array(y_augmented) 