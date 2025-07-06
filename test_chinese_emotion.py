#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试中文情感模型
"""

import json
import joblib
import numpy as np
from pathlib import Path

def test_chinese_emotion_model():
    """测试中文情感模型"""
    print("🧪 测试中文情感模型")
    print("=" * 50)
    
    # 加载模型
    model_path = Path("models/chinese_emotion")
    
    try:
        model = joblib.load(model_path / "randomforest_model.pkl")
        scaler = joblib.load(model_path / "scaler.pkl")
        label_encoder = joblib.load(model_path / "label_encoder.pkl")
        
        with open(model_path / "emotion_config.json", "r", encoding='utf-8') as f:
            emotion_config = json.load(f)
        
        print("✅ 模型加载成功")
        print(f"📊 支持的情感: {list(emotion_config.keys())}")
        print(f"🎯 标签编码器类别: {label_encoder.classes_}")
        
        # 生成测试数据
        print("\n🎵 模拟测试中文歌曲:")
        
        # 获取特征维度
        n_features = scaler.n_features_in_
        print(f"📏 特征维度: {n_features}")
        
        # 测试歌曲模拟特征
        test_songs = {
            "可惜没如果": {
                "expected": "怀念",
                "features": generate_song_features(n_features, "怀念")
            },
            "十年": {
                "expected": "怀念", 
                "features": generate_song_features(n_features, "怀念")
            },
            "小酒窝": {
                "expected": "浪漫",
                "features": generate_song_features(n_features, "浪漫")
            },
            "稻香": {
                "expected": "快乐",
                "features": generate_song_features(n_features, "快乐")
            },
            "夜曲": {
                "expected": "深沉",
                "features": generate_song_features(n_features, "深沉")
            }
        }
        
        print("\n📊 测试结果:")
        print("-" * 60)
        
        for song_name, info in test_songs.items():
            # 预测
            features_scaled = scaler.transform([info["features"]])
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            predicted_emotion = label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities) * 100
            
            # 获取情感信息
            emotion_info = emotion_config.get(predicted_emotion, {})
            icon = emotion_info.get("icon", "❓")
            
            # 显示结果
            expected = info["expected"]
            is_correct = predicted_emotion == expected
            status = "✅" if is_correct else "❌"
            
            print(f"{status} {song_name}:")
            print(f"   预测: {predicted_emotion} {icon} (置信度: {confidence:.1f}%)")
            print(f"   期望: {expected}")
            
            # 显示前3个概率
            prob_dict = dict(zip(label_encoder.classes_, probabilities))
            top_3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   前3概率: {', '.join([f'{e}({p*100:.1f}%)' for e, p in top_3])}")
            print()
        
        print("🎉 测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

def generate_song_features(n_features, emotion_type):
    """根据情感类型生成模拟特征"""
    # 基础特征
    features = np.random.normal(0, 0.1, n_features)
    
    # 根据情感类型调整特征
    emotion_templates = {
        "快乐": {"tempo": 1.2, "energy": 0.8, "valence": 0.9},
        "平静": {"tempo": 0.7, "energy": 0.3, "valence": 0.6},
        "激昂": {"tempo": 1.4, "energy": 0.9, "valence": 0.7},
        "忧郁": {"tempo": 0.6, "energy": 0.4, "valence": 0.1},
        "怀念": {"tempo": 0.75, "energy": 0.5, "valence": 0.2},
        "浪漫": {"tempo": 0.85, "energy": 0.4, "valence": 0.7},
        "深沉": {"tempo": 0.65, "energy": 0.6, "valence": 0.3},
        "激烈": {"tempo": 1.3, "energy": 0.9, "valence": 0.4}
    }
    
    template = emotion_templates.get(emotion_type, emotion_templates["平静"])
    
    # 调整特征
    if n_features >= 13:
        # MFCC特征调整
        for i in range(min(13, n_features)):
            features[i] *= template["tempo"]
    
    if n_features >= 25:
        # 色度特征调整
        for i in range(13, min(25, n_features)):
            features[i] = features[i] * template["valence"] + np.random.normal(0, 0.05)
    
    if n_features >= 35:
        # 频谱特征调整
        for i in range(25, min(35, n_features)):
            features[i] *= template["energy"]
    
    return features

if __name__ == "__main__":
    test_chinese_emotion_model() 