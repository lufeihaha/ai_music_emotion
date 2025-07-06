#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import joblib
import librosa
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, 'src')

def test_model_loading():
    """测试模型加载"""
    print("🔍 测试模型加载...")
    
    try:
        # 尝试加载模型文件
        model_paths = [
            'models/unified_results/randomforest_model.pkl',
            'models/randomforest_model.pkl',
            'models/emotion_classifier.pkl'
        ]
        
        scaler_paths = [
            'models/unified_results/scaler.pkl',
            'models/scaler.pkl'
        ]
        
        label_encoder_paths = [
            'models/unified_results/label_encoder.pkl',
            'models/label_encoder.pkl'
        ]
        
        model = None
        scaler = None
        label_encoder = None
        
        # 加载模型
        for path in model_paths:
            if os.path.exists(path):
                print(f"✅ 找到模型文件: {path}")
                model = joblib.load(path)
                print(f"✅ 模型加载成功，类型: {type(model)}")
                break
        
        # 加载标准化器
        for path in scaler_paths:
            if os.path.exists(path):
                print(f"✅ 找到标准化器文件: {path}")
                scaler = joblib.load(path)
                print(f"✅ 标准化器加载成功，类型: {type(scaler)}")
                break
        
        # 加载标签编码器
        for path in label_encoder_paths:
            if os.path.exists(path):
                print(f"✅ 找到标签编码器文件: {path}")
                label_encoder = joblib.load(path)
                print(f"✅ 标签编码器加载成功，类型: {type(label_encoder)}")
                if hasattr(label_encoder, 'classes_'):
                    print(f"✅ 情感类别: {label_encoder.classes_}")
                break
        
        if model is None:
            print("❌ 无法加载模型")
            return False
        if scaler is None:
            print("❌ 无法加载标准化器")
            return False
        if label_encoder is None:
            print("❌ 无法加载标签编码器")
            return False
            
        return True, model, scaler, label_encoder
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return False

def test_feature_extraction():
    """测试特征提取"""
    print("\n🔍 测试特征提取...")
    
    # 查找测试音频文件
    test_files = []
    for root, dirs, files in os.walk('data/processed'):
        for file in files:
            if file.endswith('.wav'):
                test_files.append(os.path.join(root, file))
                if len(test_files) >= 3:
                    break
        if len(test_files) >= 3:
            break
    
    if not test_files:
        print("❌ 找不到测试音频文件")
        return False
        
    print(f"✅ 找到 {len(test_files)} 个测试文件")
    
    try:
        # 测试特征提取
        test_file = test_files[0]
        print(f"🎵 测试文件: {test_file}")
        
        # 加载音频
        y, sr = librosa.load(test_file, sr=22050, duration=30)
        print(f"✅ 音频加载成功: 长度={len(y)}, 采样率={sr}")
        
        # 提取基本特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"✅ MFCC特征提取成功: 形状={mfcc.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征提取失败: {str(e)}")
        return False

def test_prediction():
    """测试完整预测流程"""
    print("\n🔍 测试完整预测流程...")
    
    try:
        # 加载模型组件
        result = test_model_loading()
        if not result:
            return False
            
        success, model, scaler, label_encoder = result
        if not success:
            return False
        
        # 导入预测器
        from web_app import MusicEmotionPredictor
        predictor = MusicEmotionPredictor()
        
        # 查找测试文件
        test_files = []
        for root, dirs, files in os.walk('data/processed'):
            for file in files:
                if file.endswith('.wav'):
                    test_files.append(os.path.join(root, file))
                    break
            if test_files:
                break
        
        if not test_files:
            print("❌ 找不到测试音频文件")
            return False
            
        test_file = test_files[0]
        print(f"🎵 测试预测文件: {test_file}")
        
        # 执行预测
        result = predictor.predict_emotion(test_file)
        
        if result:
            print("✅ 预测成功!")
            print(f"   预测情感: {result['predicted_emotion']}")
            print(f"   置信度: {result['confidence']:.3f}")
            print("   所有概率:")
            for emotion, prob in result['all_probabilities'].items():
                print(f"     {emotion}: {prob:.3f}")
            return True
        else:
            print("❌ 预测返回None")
            return False
            
    except Exception as e:
        print(f"❌ 预测测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 音乐情感识别系统测试")
    print("=" * 50)
    
    # 测试模型加载
    if not test_model_loading():
        sys.exit(1)
    
    # 测试特征提取
    if not test_feature_extraction():
        sys.exit(1)
        
    # 测试预测
    if not test_prediction():
        sys.exit(1)
    
    print("\n🎉 所有测试通过!")
    print("✅ 系统准备就绪，可以启动Web应用") 