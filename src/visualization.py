import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

class Visualizer:
    """可视化工具类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化工具
        
        Args:
            figsize: 图像大小
        """
        self.figsize = figsize
        plt.style.use('seaborn')
    
    def plot_waveform(self, signal: np.ndarray, sr: int, title: str = "波形图"):
        """
        绘制音频波形
        
        Args:
            signal: 音频信号
            sr: 采样率
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        librosa.display.waveshow(signal, sr=sr)
        plt.title(title)
        plt.xlabel("时间 (秒)")
        plt.ylabel("振幅")
        plt.tight_layout()
        
    def plot_spectrogram(self, signal: np.ndarray, sr: int, title: str = "频谱图"):
        """
        绘制频谱图
        
        Args:
            signal: 音频信号
            sr: 采样率
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
    
    def plot_mel_spectrogram(self, signal: np.ndarray, sr: int, title: str = "梅尔频谱图"):
        """
        绘制梅尔频谱图
        
        Args:
            signal: 音频信号
            sr: 采样率
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        mel_spect = librosa.feature.melspectrogram(y=signal, sr=sr)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                              title: str = "特征重要性"):
        """
        绘制特征重要性条形图
        
        Args:
            feature_names: 特征名称列表
            importance_scores: 重要性分数
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importance_scores
        }).sort_values('重要性', ascending=True)
        
        plt.barh(importance_df['特征'], importance_df['重要性'])
        plt.title(title)
        plt.xlabel("重要性分数")
        plt.tight_layout()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: List[str], title: str = "混淆矩阵"):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签名称
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       labels: List[str], title: str = "ROC曲线"):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            labels: 标签名称
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        
        # 二值化标签
        y_bin = label_binarize(y_true, classes=range(len(labels)))
        
        for i in range(len(labels)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            metrics: List[str], title: str = "训练历史"):
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            metrics: 要绘制的指标列表
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        
        for metric in metrics:
            if metric in history:
                plt.plot(history[metric], label=metric)
        
        plt.title(title)
        plt.xlabel('轮次')
        plt.ylabel('指标值')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    def plot_emotion_distribution(self, y: np.ndarray, labels: List[str],
                                title: str = "情感分布"):
        """
        绘制情感分布饼图
        
        Args:
            y: 标签数组
            labels: 标签名称
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        
        # 计算每个类别的样本数
        unique, counts = np.unique(y, return_counts=True)
        plt.pie(counts, labels=[labels[i] for i in unique], autopct='%1.1f%%')
        plt.title(title)
        plt.axis('equal')
        plt.tight_layout()
    
    def plot_emotion_timeline(self, emotions: List[str], probabilities: List[Dict[str, float]],
                            segment_duration: float = 3.0, title: str = "情感时间线"):
        """
        绘制情感随时间变化的堆叠面积图
        
        Args:
            emotions: 情感类别列表
            probabilities: 每个时间段的情感概率字典列表
            segment_duration: 每个片段的持续时间（秒）
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        
        # 准备数据
        time_points = np.arange(len(probabilities)) * segment_duration
        emotion_probs = np.array([[p[e] for p in probabilities] for e in emotions])
        
        # 绘制堆叠面积图
        plt.stackplot(time_points, emotion_probs, labels=emotions)
        
        plt.title(title)
        plt.xlabel("时间 (秒)")
        plt.ylabel("情感概率")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
    
    def save_figure(self, filepath: str):
        """
        保存当前图像
        
        Args:
            filepath: 保存路径
        """
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_feature_correlations(self, features: np.ndarray, feature_names: List[str],
                                title: str = "特征相关性矩阵"):
        """
        绘制特征相关性热力图
        
        Args:
            features: 特征矩阵
            feature_names: 特征名称列表
            title: 图标题
        """
        plt.figure(figsize=self.figsize)
        
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(features.T)
        
        # 绘制热力图
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=feature_names, yticklabels=feature_names)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout() 