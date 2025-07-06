#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试8情感中文模型
"""

import numpy as np
import joblib
import librosa
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ChineseEmotionTester:
    def __init__(self):
        """初始化8情感中文模型测试器"""
        self.model_dir = Path("models/comprehensive_8emotions")
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        
        # 8种中文情感配置
        self.emotions_config = {
            'happy': {'chinese': '快乐', 'icon': '😊', 'color': '#FFD700'},
            'calm': {'chinese': '平静', 'icon': '🍃', 'color': '#90EE90'},
            'energetic': {'chinese': '激昂', 'icon': '⚡', 'color': '#FF6347'},
            'melancholic': {'chinese': '忧郁', 'icon': '☁️', 'color': '#9370DB'},
            'nostalgic': {'chinese': '怀念', 'icon': '⏰', 'color': '#CD853F'},
            'romantic': {'chinese': '浪漫', 'icon': '💕', 'color': '#FF69B4'},
            'mysterious': {'chinese': '深沉', 'icon': '🌙', 'color': '#4B0082'},
            'dramatic': {'chinese': '激烈', 'icon': '🎭', 'color': '#DC143C'}
        }
        
        self.load_model()
    
    def load_model(self):
        """加载8情感模型"""
        try:
            print("🔄 加载8情感中文模型...")
            
            # 加载模型组件
            self.model = joblib.load(self.model_dir / "ensemble_model.pkl")
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            self.feature_selector = joblib.load(self.model_dir / "feature_selector.pkl")
            self.label_encoder = joblib.load(self.model_dir / "label_encoder.pkl")
            
            print("✅ 模型加载成功！")
            print(f"📊 支持的情感类别: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
        
        return True
    
    def extract_features(self, audio_file):
        """提取音频特征"""
        try:
            print(f"🎵 分析音频文件: {audio_file}")
            
            # 加载音频
            y, sr = librosa.load(audio_file, sr=22050, duration=30)
            
            # 提取特征
            features = []
            
            # 1. MFCC特征 (13维)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend([np.mean(mfcc[i]) for i in range(13)])
            features.extend([np.std(mfcc[i]) for i in range(13)])
            
            # 2. 色度特征 (12维) - 修复版本兼容性
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            except:
                # 如果chroma_stft不可用，使用替代方法
                stft = librosa.stft(y)
                chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=sr)
            
            features.extend([np.mean(chroma[i]) for i in range(12)])
            features.extend([np.std(chroma[i]) for i in range(12)])
            
            # 3. 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            
            # 4. 零交叉率
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # 5. RMS能量
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            # 6. 节拍特征
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features.append(tempo)
            except:
                features.append(120.0)  # 默认节拍
            
            # 7. 添加更多特征以匹配训练时的特征维度
            # 频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
            
            # 频谱对比度
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(min(7, spectral_contrast.shape[0])):
                features.extend([np.mean(spectral_contrast[i]), np.std(spectral_contrast[i])])
            
            # 填充到固定长度（根据实际训练时的特征维度调整）
            target_length = 200  # 增加目标长度
            while len(features) < target_length:
                features.append(0.0)
            
            return np.array(features[:target_length]).reshape(1, -1)
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_emotion(self, audio_file):
        """预测音频情感"""
        if self.model is None:
            print("❌ 模型未加载")
            return None
        
        # 提取特征
        features = self.extract_features(audio_file)
        if features is None:
            return None
        
        try:
            # 特征选择
            if self.feature_selector:
                features = self.feature_selector.transform(features)
            
            # 标准化
            if self.scaler:
                features = self.scaler.transform(features)
            
            # 预测
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # 转换为中文情感
            emotion_en = self.label_encoder.inverse_transform([prediction])[0]
            emotion_cn = self.emotions_config.get(emotion_en, {}).get('chinese', emotion_en)
            
            # 获取所有情感概率
            emotion_probs = {}
            for i, emotion_class in enumerate(self.label_encoder.classes_):
                chinese_name = self.emotions_config.get(emotion_class, {}).get('chinese', emotion_class)
                emotion_probs[chinese_name] = probabilities[i]
            
            return {
                'predicted_emotion': emotion_cn,
                'confidence': probabilities[prediction],
                'all_probabilities': emotion_probs,
                'emotion_icon': self.emotions_config.get(emotion_en, {}).get('icon', '❓'),
                'emotion_color': self.emotions_config.get(emotion_en, {}).get('color', '#808080')
            }
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def test_sample_files(self):
        """测试示例文件"""
        print("🧪 测试示例音频文件...")
        
        # 查找可用的音频文件
        audio_dirs = [
            "data/processed/calm",
            "data/processed/happy", 
            "data/processed/energetic",
            "data/processed/melancholic",
            "uploads"
        ]
        
        test_files = []
        for dir_path in audio_dirs:
            if Path(dir_path).exists():
                for ext in ['*.wav', '*.mp3']:
                    test_files.extend(list(Path(dir_path).glob(ext))[:2])  # 每个文件夹取2个
        
        if not test_files:
            print("❌ 未找到测试音频文件")
            return
        
        results = []
        for audio_file in test_files[:5]:  # 测试前5个文件
            print(f"\n📁 测试文件: {audio_file.name}")
            result = self.predict_emotion(str(audio_file))
            
            if result:
                print(f"🎯 预测情感: {result['predicted_emotion']} {result['emotion_icon']}")
                print(f"📊 置信度: {result['confidence']:.3f}")
                print(f"🎨 颜色: {result['emotion_color']}")
                
                # 显示前3个最高概率的情感
                sorted_probs = sorted(result['all_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                print("📈 概率分布:")
                for emotion, prob in sorted_probs[:3]:
                    print(f"   {emotion}: {prob:.3f}")
                
                results.append({
                    'file': audio_file.name,
                    'emotion': result['predicted_emotion'],
                    'confidence': result['confidence']
                })
        
        return results
    
    def visualize_results(self, results):
        """可视化测试结果"""
        if not results:
            return
        
        # 创建结果图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 情感分布
        emotions = [r['emotion'] for r in results]
        emotion_counts = pd.Series(emotions).value_counts()
        
        colors = [self.emotions_config.get(k, {}).get('color', '#808080') 
                 for k in emotion_counts.index]
        
        ax1.pie(emotion_counts.values, labels=emotion_counts.index, 
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('测试文件情感分布', fontsize=14)
        
        # 置信度分布
        confidences = [r['confidence'] for r in results]
        ax2.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('置信度')
        ax2.set_ylabel('文件数量')
        ax2.set_title('预测置信度分布', fontsize=14)
        ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'平均置信度: {np.mean(confidences):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('8emotions_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 结果图表已保存: 8emotions_test_results.png")

def main():
    """主函数"""
    print("🎼 8情感中文音乐情感识别模型测试")
    print("=" * 50)
    
    # 创建测试器
    tester = ChineseEmotionTester()
    
    # 测试示例文件
    results = tester.test_sample_files()
    
    if results:
        print(f"\n📋 测试完成！共测试 {len(results)} 个文件")
        
        # 可视化结果
        tester.visualize_results(results)
        
        # 打印总结
        print("\n📊 测试总结:")
        emotions_tested = set(r['emotion'] for r in results)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"   识别到的情感类型: {len(emotions_tested)} 种")
        print(f"   平均置信度: {avg_confidence:.3f}")
        print(f"   情感类型: {', '.join(emotions_tested)}")
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    main() 