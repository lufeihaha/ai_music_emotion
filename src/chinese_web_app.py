#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文音乐情感识别Web应用
支持纯中文情感分类，不映射到英文
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import joblib

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio_processor import AudioProcessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask应用配置
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-here'

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 支持的音频格式
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'aac', 'ogg'}

class ChineseEmotionPredictor:
    """中文情感预测器"""
    
    def __init__(self):
        """初始化预测器"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.emotion_config = None
        self.audio_processor = AudioProcessor()
        
        # 尝试加载中文模型
        self.load_chinese_model()
    
    def load_chinese_model(self) -> bool:
        """加载中文情感模型"""
        try:
            chinese_model_path = "models/chinese_emotion"
            
            if os.path.exists(chinese_model_path):
                # 加载中文模型
                self.model = joblib.load(f"{chinese_model_path}/randomforest_model.pkl")
                self.scaler = joblib.load(f"{chinese_model_path}/scaler.pkl")
                self.label_encoder = joblib.load(f"{chinese_model_path}/label_encoder.pkl")
                
                # 加载情感配置
                with open(f"{chinese_model_path}/emotion_config.json", "r", encoding='utf-8') as f:
                    self.emotion_config = json.load(f)
                
                logger.info("✅ 成功加载中文情感模型")
                return True
            else:
                logger.warning("❌ 中文情感模型不存在，请先训练模型")
                return False
                
        except Exception as e:
            logger.error(f"❌ 加载中文模型失败: {str(e)}")
            return False
    
    def predict_emotion(self, audio_file_path: str) -> Dict:
        """预测音频文件的情感"""
        try:
            if not self.model:
                return {
                    'success': False,
                    'error': '模型未加载，请先训练中文情感模型'
                }
            
            # 提取音频特征
            features = self.audio_processor.extract_features(audio_file_path)
            if features is None:
                return {
                    'success': False,
                    'error': '音频特征提取失败'
                }
            
            # 预处理特征
            features_scaled = self.scaler.transform([features])
            
            # 预测情感
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # 获取情感标签
            emotion = self.label_encoder.inverse_transform([prediction])[0]
            
            # 构建所有情感的概率分布
            emotion_probabilities = {}
            for i, label in enumerate(self.label_encoder.classes_):
                emotion_probabilities[label] = float(probabilities[i])
            
            # 获取情感配置
            emotion_info = self.emotion_config.get(emotion, {
                'description': '未知情感',
                'icon': '❓',
                'color': '#808080',
                'bg_color': '#F5F5F5'
            })
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': float(max(probabilities)),
                'description': emotion_info.get('description', ''),
                'icon': emotion_info.get('icon', ''),
                'color': emotion_info.get('color', '#808080'),
                'bg_color': emotion_info.get('bg_color', '#F5F5F5'),
                'all_probabilities': emotion_probabilities,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return {
                'success': False,
                'error': f'预测过程出错: {str(e)}'
            }

# 全局预测器实例
predictor = ChineseEmotionPredictor()

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """主页"""
    return render_template('chinese_emotion_app.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和情感预测"""
    try:
        # 检查是否有文件
        if 'audio_file' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            })
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            })
        
        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'不支持的文件格式。支持的格式: {", ".join(ALLOWED_EXTENSIONS)}'
            })
        
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"文件已保存: {filepath}")
        
        # 预测情感
        result = predictor.predict_emotion(filepath)
        
        # 添加文件信息
        result['filename'] = file.filename
        result['file_size'] = os.path.getsize(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"上传处理失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'文件处理失败: {str(e)}'
        })

@app.route('/model_info')
def model_info():
    """获取模型信息"""
    try:
        if not predictor.model:
            return jsonify({
                'success': False,
                'error': '模型未加载'
            })
        
        # 获取模型信息
        info = {
            'success': True,
            'model_type': '中文情感分类RandomForest模型',
            'emotions': list(predictor.emotion_config.keys()) if predictor.emotion_config else [],
            'emotion_details': predictor.emotion_config or {},
            'model_loaded': True,
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取模型信息失败: {str(e)}'
        })

@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({
        'success': False,
        'error': '文件太大，最大支持16MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """内部错误处理"""
    logger.error(f"内部错误: {str(e)}")
    return jsonify({
        'success': False,
        'error': '服务器内部错误'
    }), 500

def main():
    """主函数"""
    print("🎵 中文音乐情感识别Web应用")
    print("=" * 50)
    
    # 检查模型状态
    if predictor.model:
        print("✅ 中文情感模型已加载")
        print(f"📊 支持的情感: {list(predictor.emotion_config.keys())}")
    else:
        print("❌ 中文情感模型未加载")
        print("💡 请先运行 python src/chinese_emotion_classifier.py 训练模型")
        return
    
    # 设置模板文件夹
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
    app.template_folder = template_dir
    
    print(f"🌐 启动Web服务器...")
    print(f"📁 模板目录: {template_dir}")
    print(f"🎯 访问地址: http://localhost:5000")
    print(f"🔧 调试模式: 开启")
    
    # 启动应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

if __name__ == '__main__':
    main() 