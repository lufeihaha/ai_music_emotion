from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import librosa
from sklearn.preprocessing import StandardScaler

# 添加项目路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))

# 创建Flask应用，指定正确的模板目录
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class SimpleAudioProcessor:
    """简化的音频处理器，兼容Web应用"""
    
    def __init__(self, sr=22050, duration=30):
        self.sr = sr
        self.duration = duration
    
    def extract_features(self, file_path):
        """提取音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_max = np.max(mfccs, axis=1)
            mfcc_min = np.min(mfccs, axis=1)
            
            # 提取频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # 提取Chroma特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # 提取节奏特征
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # 提取零交叉率
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # 组合所有特征
            features = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_max, mfcc_min,
                [np.mean(spectral_centroids), np.std(spectral_centroids), 
                 np.max(spectral_centroids), np.min(spectral_centroids)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff),
                 np.max(spectral_rolloff), np.min(spectral_rolloff)],
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                 np.max(spectral_bandwidth), np.min(spectral_bandwidth)],
                chroma_mean, chroma_std,
                [tempo],
                [np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)]
            ])
            
            return features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None

class EmotionPredictor:
    """情感预测器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.emotion_mapping = {
            0: 'calm',
            1: 'energetic', 
            2: 'happy',
            3: 'melancholic'
        }
        self.load_models()
    
    def load_models(self):
        """加载训练好的模型"""
        try:
            # 查找模型文件
            models_dir = Path('../outputs/models')
            if not models_dir.exists():
                models_dir = Path('outputs/models')
            
            # 加载最佳模型 (随机森林)
            model_path = models_dir / 'random_forest.pkl'
            if model_path.exists():
                self.model = joblib.load(model_path)
                print("✓ 随机森林模型加载成功")
            
            # 加载标准化器
            scaler_path = models_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("✓ 标准化器加载成功")
            
            if self.model is None or self.scaler is None:
                print("❌ 模型或标准化器加载失败，使用默认预测")
                
        except Exception as e:
            print(f"模型加载错误: {e}")
    
    def predict(self, features):
        """预测情感"""
        if self.model is None or self.scaler is None:
            # 返回默认预测
            return {
                'emotion': 'happy',
                'probabilities': {
                    'calm': 0.2,
                    'energetic': 0.3,
                    'happy': 0.4,
                    'melancholic': 0.1
                }
            }
        
        try:
            # 标准化特征
            features_scaled = self.scaler.transform([features])
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # 构建结果
            emotion = self.emotion_mapping.get(prediction, 'unknown')
            prob_dict = {
                self.emotion_mapping.get(i, f'emotion_{i}'): float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'emotion': emotion,
                'probabilities': prob_dict
            }
            
        except Exception as e:
            print(f"预测错误: {e}")
            return {
                'emotion': 'unknown',
                'probabilities': {'unknown': 1.0}
            }

# 初始化处理器
audio_processor = SimpleAudioProcessor()
emotion_predictor = EmotionPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # 保存文件
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        try:
            # 提取特征
            features = audio_processor.extract_features(filename)
            if features is None:
                return jsonify({'error': 'Failed to process audio file'}), 400
            
            # 预测情感
            emotion_result = emotion_predictor.predict(features)
            
            # 模拟时间序列数据（实际应用中可以分段分析）
            timestamps = [0, 10, 20, 30]  # 每10秒一个时间点
            emotions = [emotion_result] * len(timestamps)  # 简化：使用相同预测
            
            # 准备返回数据
            result = {
                'timestamps': timestamps,
                'emotions': emotions,
                'filename': file.filename,
                'success': True
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
        finally:
            # 清理上传的文件
            if os.path.exists(filename):
                os.remove(filename)
    
    return jsonify({'error': 'Unknown error'}), 500

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': emotion_predictor.model is not None,
        'scaler_loaded': emotion_predictor.scaler is not None
    })

if __name__ == '__main__':
    print("🎵 音乐情感分类Web应用启动中...")
    print("📊 模型状态检查...")
    
    # 检查模型状态
    health = {
        'model_loaded': emotion_predictor.model is not None,
        'scaler_loaded': emotion_predictor.scaler is not None
    }
    
    if health['model_loaded'] and health['scaler_loaded']:
        print("✅ 所有模型加载成功")
    else:
        print("⚠️  部分模型未加载，将使用默认预测")
    
    print("🌐 启动Flask服务器...")
    print("🔗 访问地址: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 