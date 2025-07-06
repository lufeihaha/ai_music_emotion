import os
import numpy as np
import pandas as pd
import librosa
import logging
import traceback
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 特征提取参数
SR = 22050  # 采样率
DURATION = 30  # 音频片段长度(秒)
HOP_LENGTH = 512  # STFT的hop length
N_MELS = 128  # 梅尔频谱的频带数
N_MFCC = 20  # MFCC系数数量
FRAME_LENGTH = 2048  # 帧长度

def extract_features(audio_path):
    """
    从音频文件中提取特征
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            logging.error(f"文件不存在: {audio_path}")
            return None
            
        # 加载音频文件
        try:
            y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
        except Exception as e:
            logging.error(f"加载音频文件失败 {audio_path}: {str(e)}")
            return None
        
        # 1. 提取MFCC特征
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            mfcc_mean = np.mean(mfcc, axis=1)  # (20,)
            mfcc_std = np.std(mfcc, axis=1)    # (20,)
        except Exception as e:
            logging.error(f"提取MFCC特征失败 {audio_path}: {str(e)}")
            return None
        
        # 2. 提取频谱特征
        try:
            # 频谱质心
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroids_mean = np.mean(spectral_centroids)
            spectral_centroids_std = np.std(spectral_centroids)
            
            # 频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            spectral_bandwidth_std = np.std(spectral_bandwidth)
            
            # 频谱衰减
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            spectral_rolloff_std = np.std(spectral_rolloff)
            
            # 频谱对比度
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            spectral_contrast_std = np.std(spectral_contrast, axis=1)
        except Exception as e:
            logging.error(f"提取频谱特征失败 {audio_path}: {str(e)}")
            return None
        
        # 3. 提取色度特征
        try:
            # 常规色度特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)  # (12,)
            chroma_std = np.std(chroma, axis=1)    # (12,)
            
            # CQT色度图
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_cqt_mean = np.mean(chroma_cqt, axis=1)
            chroma_cqt_std = np.std(chroma_cqt, axis=1)
        except Exception as e:
            logging.error(f"提取色度特征失败 {audio_path}: {str(e)}")
            return None
        
        # 4. 提取时域特征
        try:
            # 零交叉率
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            zero_crossing_rate_mean = np.mean(zero_crossing_rate)
            zero_crossing_rate_std = np.std(zero_crossing_rate)
            
            # RMS能量
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            
            # 短时能量
            frame_length = FRAME_LENGTH
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=HOP_LENGTH)
            energy = np.sum(frames**2, axis=0) / frame_length
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
        except Exception as e:
            logging.error(f"提取时域特征失败 {audio_path}: {str(e)}")
            return None
        
        # 5. 提取节奏特征
        try:
            # 节拍强度和节拍
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # 节拍模式
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            pulse_mean = np.mean(pulse)
            pulse_std = np.std(pulse)
        except Exception as e:
            logging.error(f"提取节奏特征失败 {audio_path}: {str(e)}")
            return None
        
        # 6. 提取深度学习特征
        try:
            # 梅尔频谱图
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
            mel_spec_mean = np.mean(mel_spec, axis=1)
            mel_spec_std = np.std(mel_spec, axis=1)
            
            # 常数Q变换
            C = np.abs(librosa.cqt(y=y, sr=sr))
            cqt_mean = np.mean(C, axis=1)
            cqt_std = np.std(C, axis=1)
        except Exception as e:
            logging.error(f"提取深度学习特征失败 {audio_path}: {str(e)}")
            return None
        
        # 合并所有特征
        try:
            features = np.concatenate([
                # MFCC特征
                mfcc_mean.ravel(),                          # (20,)
                mfcc_std.ravel(),                           # (20,)
                
                # 频谱特征
                np.array([spectral_centroids_mean]).ravel(),  # (1,)
                np.array([spectral_centroids_std]).ravel(),   # (1,)
                np.array([spectral_bandwidth_mean]).ravel(),  # (1,)
                np.array([spectral_bandwidth_std]).ravel(),   # (1,)
                np.array([spectral_rolloff_mean]).ravel(),    # (1,)
                np.array([spectral_rolloff_std]).ravel(),     # (1,)
                spectral_contrast_mean.ravel(),             # (7,)
                spectral_contrast_std.ravel(),              # (7,)
                
                # 色度特征
                chroma_mean.ravel(),                        # (12,)
                chroma_std.ravel(),                         # (12,)
                chroma_cqt_mean.ravel(),                    # (12,)
                chroma_cqt_std.ravel(),                     # (12,)
                
                # 时域特征
                np.array([zero_crossing_rate_mean]).ravel(),  # (1,)
                np.array([zero_crossing_rate_std]).ravel(),   # (1,)
                np.array([rms_mean]).ravel(),                # (1,)
                np.array([rms_std]).ravel(),                 # (1,)
                np.array([energy_mean]).ravel(),             # (1,)
                np.array([energy_std]).ravel(),              # (1,)
                
                # 节奏特征
                np.array([tempo]).ravel(),                   # (1,)
                np.array([pulse_mean]).ravel(),              # (1,)
                np.array([pulse_std]).ravel(),               # (1,)
                
                # 深度学习特征
                mel_spec_mean.ravel()[:20],                # (20,)
                mel_spec_std.ravel()[:20],                 # (20,)
                cqt_mean.ravel()[:20],                     # (20,)
                cqt_std.ravel()[:20]                       # (20,)
            ])
            return features
        except Exception as e:
            logging.error(f"合并特征失败 {audio_path}: {str(e)}")
            return None
    
    except Exception as e:
        logging.error(f"处理文件 {audio_path} 时出错:\n{traceback.format_exc()}")
        return None

def main():
    # 设置路径
    processed_dir = "data/processed"
    features_file = "data/features.npy"
    labels_file = os.path.join(processed_dir, "emotion_labels.csv")
    
    # 读取标签文件
    try:
        labels_df = pd.read_csv(labels_file)
        logging.info(f"成功读取标签文件，共 {len(labels_df)} 个样本")
    except Exception as e:
        logging.error(f"读取标签文件失败: {str(e)}")
        return
    
    # 初始化特征列表
    features_list = []
    valid_indices = []
    
    # 遍历所有音频文件并提取特征
    logging.info("开始提取特征...")
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        try:
            audio_path = os.path.join(processed_dir, row['file_path'])
            features = extract_features(audio_path)
            
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
        except Exception as e:
            logging.error(f"处理样本 {idx} 时出错: {str(e)}")
            continue
    
    if not features_list:
        logging.error("没有成功提取任何特征")
        return
    
    # 转换为numpy数组
    try:
        features_array = np.array(features_list)
        logging.info(f"特征数组形状: {features_array.shape}")
    except Exception as e:
        logging.error(f"转换特征数组失败: {str(e)}")
        return
    
    # 保存特征
    try:
        np.save(features_file, features_array)
        logging.info(f"特征已保存到 {features_file}")
    except Exception as e:
        logging.error(f"保存特征失败: {str(e)}")
        return
    
    # 更新标签文件，只保留成功提取特征的样本
    try:
        valid_labels_df = labels_df.iloc[valid_indices]
        valid_labels_df.to_csv(labels_file, index=False)
        logging.info(f"更新后的标签文件已保存，保留了 {len(valid_labels_df)} 个样本")
    except Exception as e:
        logging.error(f"更新标签文件失败: {str(e)}")
        return
    
    logging.info(f"特征提取完成。共处理 {len(features_list)} 个文件。")
    logging.info(f"特征形状: {features_array.shape}")

if __name__ == "__main__":
    main() 