import os
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt
from audio_processor import AudioProcessor
import logging
from typing import Optional, Tuple, List

logging.basicConfig(level=logging.INFO)

class EmotionPredictor:
    def __init__(self, model_dir: str):
        """
        初始化情感预测器
        
        Args:
            model_dir: 模型目录路径
        """
        self.model = joblib.load(os.path.join(model_dir, 'music_emotion_classifier.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        self.processor = AudioProcessor(sr=22050, duration=30)
        
    def predict_emotion(self, audio_path: str, visualize: bool = True) -> dict:
        """
        预测音频文件的情感
        
        Args:
            audio_path: 音频文件路径
            visualize: 是否可视化音频特征
            
        Returns:
            包含预测结果的字典
        """
        try:
            # 提取特征
            features = self.processor.process_audio_file(audio_path)
            features = features.reshape(1, -1)
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 预测情感
            emotion_pred = self.model.predict(features_scaled)[0]
            emotion_probs = self.model.predict_proba(features_scaled)[0]
            
            # 获取所有情感标签的概率
            emotion_probabilities = {
                emotion: prob
                for emotion, prob in zip(self.label_encoder.classes_, emotion_probs)
            }
            
            # 获取预测的情感标签
            predicted_emotion = self.label_encoder.inverse_transform([emotion_pred])[0]
            
            # 如果需要可视化
            if visualize:
                self._visualize_prediction(audio_path, emotion_probabilities)
            
            return {
                'predicted_emotion': predicted_emotion,
                'confidence': emotion_probabilities[predicted_emotion],
                'all_probabilities': emotion_probabilities
            }
            
        except Exception as e:
            logging.error(f"预测过程中出错: {str(e)}")
            raise
            
    def _visualize_prediction(self, audio_path: str, probabilities: dict):
        """可视化预测结果和音频特征"""
        # 创建一个2x2的图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 波形图
        y, sr = librosa.load(audio_path, duration=30)
        times = np.arange(len(y)) / sr
        ax1.plot(times, y)
        ax1.set_title('波形图')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('振幅')
        
        # 2. 梅尔频谱图
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_title('梅尔频谱图')
        
        # 3. 色度图
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax3)
        ax3.set_title('色度图')
        
        # 4. 情感概率条形图
        emotions = list(probabilities.keys())
        probs = list(probabilities.values())
        ax4.bar(emotions, probs)
        ax4.set_title('情感预测概率')
        ax4.set_ylim(0, 1)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = os.path.join(os.path.dirname(audio_path), 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 
                                 f"{os.path.splitext(os.path.basename(audio_path))[0]}_analysis.png")
        plt.savefig(output_path)
        plt.close()
        
        logging.info(f"分析图表已保存到: {output_path}")

def load_audio_file(file_path: str) -> Optional[np.ndarray]:
    """加载音频文件并提取特征"""
    try:
        # 加载音频文件
        y, sr = librosa.load(file_path, duration=30)
        
        # 提取特征
        # 1. MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. 频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # 3. 节奏特征
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 4. 和声特征
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # 转换为一维
        
        # 5. 能量特征
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # 组合所有特征
        feature_list = [
            mfcc_mean,  # (20,)
            mfcc_std,   # (20,)
            np.array([np.mean(spectral_centroids), np.std(spectral_centroids)]),  # (2,)
            np.array([np.mean(spectral_rolloff), np.std(spectral_rolloff)]),      # (2,)
            np.array([tempo]),  # (1,)
            chroma_mean,  # (12,)
            np.array([rms_mean, rms_std])  # (2,)
        ]
        
        # 确保所有特征都是一维的
        features = np.concatenate([np.ravel(f) for f in feature_list])
        return features
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def predict_emotion(model_path: str, scaler_path: str, audio_file: str) -> Tuple[str, List[Tuple[str, float]]]:
    """预测音乐文件的情感"""
    try:
        # 加载模型和标准化器
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # 提取特征
        features = load_audio_file(audio_file)
        if features is None:
            raise ValueError("特征提取失败")
            
        # 标准化特征
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # 预测情感
        emotion = model.predict(features_scaled)[0]
        
        # 获取各个类别的概率
        proba = model.predict_proba(features_scaled)[0]
        emotion_probs = list(zip(model.classes_, proba))
        emotion_probs.sort(key=lambda x: x[1], reverse=True)
        
        return emotion, emotion_probs
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        raise

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    model_path = os.path.join(project_dir, "models", "emotion_classifier.pkl")
    scaler_path = os.path.join(project_dir, "models", "scaler.pkl")
    
    # 获取音频文件路径
    audio_file = input("请输入音频文件路径: ")
    if not os.path.exists(audio_file):
        print(f"找不到文件: {audio_file}")
        return
    
    try:
        # 预测情感
        emotion, probs = predict_emotion(model_path, scaler_path, audio_file)
        
        # 输出结果
        print(f"\n预测结果: {emotion}")
        print("\n各情感类别的概率:")
        for emotion_class, prob in probs:
            print(f"{emotion_class}: {prob:.2%}")
            
    except Exception as e:
        print(f"预测失败: {str(e)}")

if __name__ == "__main__":
    main() 