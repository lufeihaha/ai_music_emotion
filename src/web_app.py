#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import librosa
from flask import Flask, request, render_template, jsonify, send_file
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
from pathlib import Path

# 导入中文字体配置
from matplotlib_chinese_config import setup_chinese_matplotlib
setup_chinese_matplotlib()

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置日志
logging.basicConfig(level=logging.INFO)

# 创建Flask应用，设置正确的模板目录
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
print(f"🔍 当前文件路径: {__file__}")
print(f"🔍 模板目录路径: {template_dir}")
print(f"🔍 模板目录存在: {os.path.exists(template_dir)}")
print(f"🔍 music_emotion_app.html存在: {os.path.exists(os.path.join(template_dir, 'music_emotion_app.html'))}")

app = Flask(__name__, template_folder=template_dir)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size for batch processing

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class MusicEmotionPredictor:
    def __init__(self):
        """初始化预测器"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.confidence_calibrator = None
        self.model_type = None
        self.emotions_config = None
        
        # 尝试按优先级加载模型 - 优先使用最高性能模型
        if self._load_optimized_enhanced_model():
            logging.info("🎉 使用优化增强模型 (80%+准确率) - 终极性能")
            self.model_type = "optimized_enhanced"
        elif self._load_enhanced_emotion_model():
            logging.info("✅ 使用增强情感模型 (77%准确率) - 最佳性能")
            self.model_type = "enhanced_emotion"
        elif self._load_improved_targeted_model():
            logging.info("✅ 加载改进的针对性模型成功 (67.5%准确率)")
            self.model_type = "improved_targeted"
        elif self._load_comprehensive_8emotions_model():
            logging.info("✅ 加载综合8种情感模型成功 (46.3%准确率)")
            self.model_type = "comprehensive_8emotions"
        elif self._load_chinese_optimized_model():
            logging.info("使用中文优化模型")
            self.model_type = "chinese_optimized"
        else:
            self._load_original_model()
            logging.info("使用原始模型")
            self.model_type = "original"
    
    def _load_optimized_enhanced_model(self) -> bool:
        """加载优化增强模型 - 最高优先级 (80%+准确率)"""
        try:
            model_dir = "models/optimized_enhanced"
            if not os.path.exists(model_dir):
                return False
            
            model_path = f"{model_dir}/enhanced_model.pkl"
            scaler_path = f"{model_dir}/scaler.pkl"
            selector_path = f"{model_dir}/feature_selector.pkl"
            
            if all(os.path.exists(p) for p in [model_path, scaler_path, selector_path]):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.feature_selector = joblib.load(selector_path)
                
                # 检查性能报告
                report_path = f"{model_dir}/performance_report.json"
                if os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        import json
                        report = json.load(f)
                        accuracy = report.get('test_accuracy', 0)
                        logging.info(f"   模型性能: {accuracy*100:.2f}%准确率")
                
                logging.info("🎉 加载优化增强模型成功")
                return True
            return False
        except Exception as e:
            logging.error(f"❌ 加载优化增强模型失败: {str(e)}")
            return False
    
    def _load_comprehensive_8emotions_model(self):
        """加载综合8种情感模型"""
        try:
            model_dir = Path("models/comprehensive_8emotions")
            
            if not all([
                (model_dir / "ensemble_model.pkl").exists(),
                (model_dir / "scaler.pkl").exists(),
                (model_dir / "label_encoder.pkl").exists(),
                (model_dir / "feature_selector.pkl").exists()
            ]):
                return False
            
            # 加载模型组件
            self.model = joblib.load(model_dir / "ensemble_model.pkl")
            self.scaler = joblib.load(model_dir / "scaler.pkl")
            self.label_encoder = joblib.load(model_dir / "label_encoder.pkl")
            self.feature_selector = joblib.load(model_dir / "feature_selector.pkl")
            
            # 加载情感配置
            config_path = model_dir / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    import json
                    config = json.load(f)
                    
                # 8种情感配置
                self.emotions_config = {
                    'happy': {'icon': '😊', 'chinese': '快乐', 'color': '#FFD700'},
                    'calm': {'icon': '🍃', 'chinese': '平静', 'color': '#90EE90'},
                    'energetic': {'icon': '⚡', 'chinese': '激昂', 'color': '#FF6347'},
                    'melancholic': {'icon': '☁️', 'chinese': '忧郁', 'color': '#9370DB'},
                    'nostalgic': {'icon': '⏰', 'chinese': '怀念', 'color': '#CD853F'},
                    'romantic': {'icon': '💕', 'chinese': '浪漫', 'color': '#FF69B4'},
                    'mysterious': {'icon': '🌙', 'chinese': '深沉', 'color': '#4B0082'},
                    'dramatic': {'icon': '🎭', 'chinese': '激烈', 'color': '#DC143C'}
                }
            
            return True
            
        except Exception as e:
            logging.error(f"加载综合8种情感模型失败: {e}")
            return False
    
    def _load_improved_targeted_model(self) -> bool:
        """加载改进的针对性模型 - 最高优先级"""
        try:
            model_dir = "models/improved_targeted"
            if not os.path.exists(model_dir):
                return False
            
            model_path = f"{model_dir}/ensemble_model.pkl"
            scaler_path = f"{model_dir}/scaler.pkl"
            encoder_path = f"{model_dir}/label_encoder.pkl"
            selector_path = f"{model_dir}/feature_selector.pkl"
            calibrator_path = f"{model_dir}/confidence_calibrator.pkl"
            
            if all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoder = joblib.load(encoder_path)
                
                # 加载可选组件
                if os.path.exists(selector_path):
                    self.feature_selector = joblib.load(selector_path)
                if os.path.exists(calibrator_path):
                    self.confidence_calibrator = joblib.load(calibrator_path)
                
                logging.info("✅ 加载改进的针对性模型成功")
                return True
            return False
        except Exception as e:
            logging.error(f"❌ 加载改进的针对性模型失败: {str(e)}")
            return False
    
    def _load_enhanced_emotion_model(self) -> bool:
        """加载增强情感模型 - 第二优先级"""
        try:
            model_dir = "models/enhanced_emotion"
            if not os.path.exists(model_dir):
                return False
            
            model_path = f"{model_dir}/model.pkl"
            scaler_path = f"{model_dir}/scaler.pkl"
            encoder_path = f"{model_dir}/label_encoder.pkl"
            
            if all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoder = joblib.load(encoder_path)
                
                logging.info("✅ 加载增强情感模型成功")
                return True
            return False
        except Exception as e:
            logging.error(f"❌ 加载增强情感模型失败: {str(e)}")
            return False
    
    def _load_chinese_optimized_model(self) -> bool:
        """尝试加载中文优化模型 - 第三优先级"""
        # 优先尝试加载纯中文情感模型
        try:
            chinese_emotion_path = "models/chinese_emotion"
            if os.path.exists(chinese_emotion_path):
                model_path = f"{chinese_emotion_path}/randomforest_model.pkl"
                scaler_path = f"{chinese_emotion_path}/scaler.pkl"
                encoder_path = f"{chinese_emotion_path}/label_encoder.pkl"
                config_path = f"{chinese_emotion_path}/emotion_config.json"
                
                if all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path, config_path]):
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    self.label_encoder = joblib.load(encoder_path)
                    
                    # 加载中文情感配置
                    with open(config_path, "r", encoding='utf-8') as f:
                        self.emotion_config = json.load(f)
                    
                    logging.info("✅ 加载纯中文情感模型成功")
                    return True
        except Exception as e:
            logging.warning(f"⚠️ 加载纯中文情感模型失败: {str(e)}")
        
        # 回退到中文优化模型
        try:
            model_path = "models/chinese_optimized/randomforest_model.pkl"
            scaler_path = "models/chinese_optimized/scaler.pkl"
            encoder_path = "models/chinese_optimized/label_encoder.pkl"
            
            if all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoder = joblib.load(encoder_path)
                logging.info("✅ 加载中文优化模型成功")
                return True
            return False
        except Exception as e:
            logging.error(f"❌ 加载中文优化模型失败: {str(e)}")
            return False
    
    def _load_original_model(self):
        """加载原始模型"""
        try:
            model_path = "models/unified_results/randomforest_model.pkl"
            scaler_path = "models/unified_results/scaler.pkl"
            encoder_path = "models/unified_results/label_encoder.pkl"
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            logging.info("Loaded original model")
        except Exception as e:
            logging.error(f"Failed to load original model: {str(e)}")
            raise
    
    def extract_features(self, audio_path, sr=22050, duration=30):
        """提取音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
            
            # 分段处理
            segment_length = len(y) // 128  # 128个片段
            features_list = []
            
            for i in range(128):
                start = i * segment_length
                end = (i + 1) * segment_length
                segment = y[start:end] if end <= len(y) else y[start:]
                
                if len(segment) > 0:
                    # 提取基本特征
                    segment_features = self.extract_segment_features(segment, sr)
                    features_list.append(segment_features)
            
            # 转换为numpy数组并重塑
            features = np.array(features_list)
            if len(features.shape) == 2:
                features = features.flatten()
            
            return features
            
        except Exception as e:
            logging.error(f"Feature extraction error: {str(e)}")
            return None
    
    def extract_segment_features(self, segment, sr):
        """提取单个片段的特征"""
        features = []
        
        try:
            # MFCC特征
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1).flatten()
            mfcc_std = np.std(mfcc, axis=1).flatten()
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
            
            features.extend([
                float(np.mean(spectral_centroids)),
                float(np.std(spectral_centroids)),
                float(np.mean(spectral_rolloff)),
                float(np.std(spectral_rolloff))
            ])
            
            # 色度特征
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            chroma_mean = np.mean(chroma, axis=1).flatten()
            chroma_std = np.std(chroma, axis=1).flatten()
            features.extend(chroma_mean)
            features.extend(chroma_std)
            
            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(segment)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # RMS能量
            rms = librosa.feature.rms(y=segment)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
        except Exception as e:
            logging.warning(f"Error in segment feature extraction: {str(e)}")
            features = [[0] * 38]  # 默认特征大小，包装为列表
        
        # 确保特征长度一致
        features_flat = []
        for f in features:
            if hasattr(f, '__iter__') and not isinstance(f, str):
                features_flat.extend(f)
            else:
                features_flat.append(f)
        
        # 填充或截断到固定长度
        target_length = 38
        if len(features_flat) > target_length:
            features_flat = features_flat[:target_length]
        elif len(features_flat) < target_length:
            features_flat.extend([0] * (target_length - len(features_flat)))
        
        return np.array(features_flat)
    
    def predict_emotion(self, audio_path):
        """预测音频情感"""
        try:
            # 检查模型是否已加载
            if self.model is None or self.scaler is None or self.label_encoder is None:
                logging.error("模型组件未正确加载")
                return None
            
            # 提取特征
            features = self.extract_features(audio_path)
            if features is None:
                return None

            # 重塑特征以匹配训练时的格式
            features = features.reshape(1, -1)
            
            # 根据模型类型进行不同的处理
            if self.model_type == "comprehensive_8emotions":
                # 8种情感模型：先标准化，再特征选择
                features_scaled = self.scaler.transform(features)
                if self.feature_selector is not None:
                    features_scaled = self.feature_selector.transform(features_scaled)
            elif self.model_type == "improved_targeted":
                # 改进模型：先特征选择，再标准化
                if self.feature_selector is not None:
                    features_selected = self.feature_selector.transform(features)
                    features_scaled = self.scaler.transform(features_selected)
                else:
                    features_scaled = self.scaler.transform(features)
            else:
                # 其他模型：直接标准化
                features_scaled = self.scaler.transform(features)

            # 预测
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]

            # 转换标签
            emotion = self.label_encoder.inverse_transform([prediction])[0]

            # 获取所有情感的概率
            emotion_probs = {}
            emotion_classes = self.label_encoder.classes_
            
            for i, emotion_class in enumerate(emotion_classes):
                # 如果有8种情感配置，使用中文名称
                if self.emotions_config and emotion_class in self.emotions_config:
                    chinese_name = self.emotions_config[emotion_class]['chinese']
                    emotion_probs[chinese_name] = float(probabilities[i])
                else:
                    # 使用原始英文名称或转换为中文
                    chinese_mapping = {
                        'happy': '快乐',
                        'calm': '平静', 
                        'energetic': '激昂',
                        'melancholic': '忧郁',
                        'nostalgic': '怀念',
                        'romantic': '浪漫',
                        'mysterious': '深沉',
                        'dramatic': '激烈'
                    }
                    chinese_name = chinese_mapping.get(emotion_class, emotion_class)
                    emotion_probs[chinese_name] = float(probabilities[i])

            # 转换主要预测结果为中文
            if self.emotions_config and emotion in self.emotions_config:
                predicted_chinese = self.emotions_config[emotion]['chinese']
            else:
                chinese_mapping = {
                    'happy': '快乐',
                    'calm': '平静',
                    'energetic': '激昂', 
                    'melancholic': '忧郁',
                    'nostalgic': '怀念',
                    'romantic': '浪漫',
                    'mysterious': '深沉',
                    'dramatic': '激烈'
                }
                predicted_chinese = chinese_mapping.get(emotion, emotion)

            # 计算置信度
            confidence = float(np.max(probabilities))

            return {
                'predicted_emotion': predicted_chinese,
                'confidence': confidence,
                'all_probabilities': emotion_probs,
                'model_type': self.model_type,
                'emotion_count': len(emotion_classes)
            }

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None

    def get_emotion_icon_and_color(self, emotion):
        """获取情感对应的图标和颜色"""
        # 如果有中文情感配置，优先使用
        if hasattr(self, 'emotion_config') and self.emotion_config and emotion in self.emotion_config:
            config = self.emotion_config[emotion]
            return {
                'icon': config.get('icon', '❓'),
                'color': config.get('color', '#808080'),
                'bg_color': config.get('bg_color', '#F5F5F5')
            }
        
        # 回退到默认配置
        emotion_styles = {
            'happy': {'icon': '😊', 'color': '#FFD700', 'bg_color': '#FFF8DC'},
            'calm': {'icon': '🍃', 'color': '#90EE90', 'bg_color': '#F0FFF0'},
            'energetic': {'icon': '⚡', 'color': '#FF6347', 'bg_color': '#FFF0F5'},
            'melancholic': {'icon': '☁️', 'color': '#9370DB', 'bg_color': '#F8F8FF'},
            # 新增的中文优化情感类别
            'nostalgic': {'icon': '⏰', 'color': '#CD853F', 'bg_color': '#FDF5E6'},
            'romantic': {'icon': '💕', 'color': '#FF69B4', 'bg_color': '#FFF0F5'},
            'dramatic': {'icon': '🎭', 'color': '#DC143C', 'bg_color': '#FFF8F8'},
            'mysterious': {'icon': '🌙', 'color': '#4B0082', 'bg_color': '#F5F5F5'},
            # 中文情感
            '快乐': {'icon': '😊', 'color': '#FFD700', 'bg_color': '#FFF8DC'},
            '平静': {'icon': '🍃', 'color': '#90EE90', 'bg_color': '#F0FFF0'},
            '激昂': {'icon': '⚡', 'color': '#FF6347', 'bg_color': '#FFF0F5'},
            '忧郁': {'icon': '☁️', 'color': '#9370DB', 'bg_color': '#F8F8FF'},
            '怀念': {'icon': '⏰', 'color': '#CD853F', 'bg_color': '#FDF5E6'},
            '浪漫': {'icon': '💕', 'color': '#FF69B4', 'bg_color': '#FFF0F5'},
            '深沉': {'icon': '🌙', 'color': '#4B0082', 'bg_color': '#F5F5F5'},
            '激烈': {'icon': '🎭', 'color': '#DC143C', 'bg_color': '#FFF8F8'}
        }
        return emotion_styles.get(emotion, emotion_styles.get('calm', {'icon': '❓', 'color': '#808080', 'bg_color': '#F5F5F5'}))

    def get_emotion_description(self, emotion):
        """获取情感的中文描述"""
        # 如果有中文情感配置，优先使用
        if hasattr(self, 'emotion_config') and self.emotion_config and emotion in self.emotion_config:
            return self.emotion_config[emotion].get('description', emotion)
        
        # 回退到默认描述
        descriptions = {
            'happy': '快乐',
            'calm': '平静',
            'energetic': '充满活力',
            'melancholic': '忧郁',
            # 新增的中文优化情感描述
            'nostalgic': '怀念',
            'romantic': '浪漫',
            'dramatic': '戏剧性',
            'mysterious': '神秘',
            # 中文情感直接返回
            '快乐': '欢快、愉悦、开心的情感',
            '平静': '宁静、安详、放松的情感',
            '激昂': '激动、热情、充满活力的情感',
            '忧郁': '悲伤、忧愁、低沉的情感',
            '怀念': '思念、回忆、眷恋的情感',
            '浪漫': '温柔、甜蜜、爱情的情感',
            '深沉': '深刻、沉重、内敛的情感',
            '激烈': '强烈、紧张、戏剧性的情感'
        }
        return descriptions.get(emotion, emotion)

# 创建预测器实例
try:
    predictor = MusicEmotionPredictor()
    logging.info("Predictor initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize predictor: {str(e)}")
    predictor = None

@app.route('/')
def index():
    """主页面"""
    return render_template('music_emotion_app.html')

@app.route('/apple')
def apple_index():
    """苹果风格主页面"""
    return render_template('apple_style_music_emotion_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理音频预测请求"""
    logging.info("=== 开始处理预测请求 ===")
    
    if predictor is None:
        logging.error("预测器未初始化")
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    if 'audio' not in request.files:
        logging.error("请求中没有音频文件")
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        logging.error("没有选择文件")
        return jsonify({'error': 'No file selected'}), 400
    
    logging.info(f"接收到文件: {file.filename}, 大小: {file.content_length if hasattr(file, 'content_length') else 'unknown'}")
    
    if file and file.filename and allowed_file(file.filename):
        try:
            # 保存上传的文件
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            logging.info(f"保存文件到: {filepath}")
            file.save(filepath)
            
            # 检查文件是否保存成功
            if not os.path.exists(filepath):
                logging.error(f"文件保存失败: {filepath}")
                return jsonify({'error': 'Failed to save file'}), 500
                
            logging.info(f"文件保存成功，大小: {os.path.getsize(filepath)} bytes")
            
            # 预测情感
            logging.info("开始预测情感...")
            result = predictor.predict_emotion(filepath)
            
            # 清理临时文件
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.info("临时文件已清理")
            
            if result:
                logging.info(f"预测成功: {result}")
                return jsonify(result)
            else:
                logging.error("预测返回None")
                return jsonify({'error': 'Prediction failed'}), 500
                
        except Exception as e:
            logging.error(f"预测端点错误: {str(e)}", exc_info=True)
            # 确保清理临时文件
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    logging.error(f"文件格式不支持: {file.filename}")
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/visualize', methods=['POST'])
def visualize():
    """生成可视化图表"""
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file and file.filename and allowed_file(file.filename):
        try:
            # 保存文件
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 加载音频进行可视化
            y, sr = librosa.load(filepath, duration=30)
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Audio Analysis', fontsize=16)
            
            # 波形图
            times = np.arange(len(y)) / sr
            axes[0, 0].plot(times, y)
            axes[0, 0].set_title('Waveform')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            
            # 频谱图
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0, 1])
            axes[0, 1].set_title('Spectrogram')
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1, 0])
            axes[1, 0].set_title('MFCC')
            
            # 色度图
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[1, 1])
            axes[1, 1].set_title('Chromagram')
            
            plt.tight_layout()
            
            # 转换为base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            # 清理文件
            os.remove(filepath)
            
            return jsonify({'image': img_str})
            
        except Exception as e:
            logging.error(f"Visualization error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """收集用户反馈"""
    try:
        data = request.get_json()
        
        feedback_data = {
            'file_name': data.get('file_name'),
            'predicted_emotion': data.get('predicted_emotion'),
            'actual_emotion': data.get('actual_emotion'),
            'confidence': data.get('confidence'),
            'user_rating': data.get('user_rating', 0),
            'user_comment': data.get('user_comment', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存反馈到文件
        feedback_file = 'data/user_feedback.jsonl'
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        
        logging.info(f"User feedback collected: {feedback_data}")
        return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})
        
    except Exception as e:
        logging.error(f"Failed to collect feedback: {str(e)}")
        return jsonify({'error': 'Failed to save feedback'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测多个音频文件"""
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    if 'audio_files' not in request.files:
        return jsonify({'error': 'No audio files provided'}), 400
    
    files = request.files.getlist('audio_files')
    
    # 检查文件数量限制
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 files allowed for batch processing'}), 400
    
    # 检查总文件大小
    total_size = 0
    for file in files:
        if hasattr(file, 'content_length') and file.content_length:
            total_size += file.content_length
    
    if total_size > 80 * 1024 * 1024:  # 80MB limit for batch
        return jsonify({'error': 'Total file size exceeds 80MB limit'}), 400
    
    results = []
    processed_count = 0
    
    for i, file in enumerate(files):
        if file and file.filename and allowed_file(file.filename):
            try:
                # 检查单个文件大小
                file.seek(0, 2)  # 移动到文件末尾
                file_size = file.tell()
                file.seek(0)  # 重置到开头
                
                if file_size > 20 * 1024 * 1024:  # 20MB per file limit
                    results.append({
                        'original_filename': file.filename,
                        'error': f'File size ({file_size/1024/1024:.1f}MB) exceeds 20MB limit'
                    })
                    continue
                
                # 保存文件
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{i}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                logging.info(f"Processing batch file {i+1}/{len(files)}: {file.filename}")
                file.save(filepath)
                
                # 预测
                result = predictor.predict_emotion(filepath)
                if result:
                    result['original_filename'] = file.filename
                    result['file_size_mb'] = round(file_size / 1024 / 1024, 2)
                    results.append(result)
                    processed_count += 1
                else:
                    results.append({
                        'original_filename': file.filename,
                        'error': 'Prediction failed'
                    })
                
                # 清理文件
                if os.path.exists(filepath):
                    os.remove(filepath)
                
            except Exception as e:
                logging.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    'original_filename': file.filename,
                    'error': str(e)
                })
                
                # 清理可能存在的文件
                if 'filepath' in locals() and os.path.exists(filepath):
                    os.remove(filepath)
        else:
            results.append({
                'original_filename': file.filename if file.filename else 'Unknown',
                'error': 'Invalid file format or empty file'
            })
    
    return jsonify({
        'results': results, 
        'total_processed': processed_count,
        'total_files': len(files),
        'success_rate': f"{(processed_count/len(files)*100):.1f}%" if files else "0%"
    })

@app.route('/history')
def get_history():
    """获取分析历史记录"""
    try:
        history_file = 'data/analysis_history.jsonl'
        if not os.path.exists(history_file):
            return jsonify({'history': []})
        
        history = []
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    history.append(json.loads(line))
                except:
                    continue
        
        # 返回最近的50条记录
        return jsonify({'history': history[-50:]})
        
    except Exception as e:
        logging.error(f"Failed to get history: {str(e)}")
        return jsonify({'error': 'Failed to get history'}), 500

@app.route('/health')
def health():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'predictor_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

def allowed_file(filename):
    """检查文件类型是否允许"""
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    print("🎵 启动音乐情感识别Web应用...")
    print("📊 模型状态:", "✅ 已加载" if predictor is not None else "❌ 未加载")
    print("🌐 访问地址: http://localhost:5000")
    print("🔧 调试模式: 已启用")
    app.run(debug=True, host='0.0.0.0', port=5000) 