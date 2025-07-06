#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的8情感中文模型测试
"""

import numpy as np
import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# 添加src目录到路径
sys.path.append('src')

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_8emotions_model():
    """测试8情感中文模型"""
    print("🎼 8情感中文音乐情感识别模型测试")
    print("=" * 50)
    
    # 模型路径
    model_dir = Path("models/comprehensive_8emotions")
    
    # 检查模型文件
    required_files = [
        "ensemble_model.pkl",
        "scaler.pkl", 
        "feature_selector.pkl",
        "label_encoder.pkl"
    ]
    
    print("🔍 检查模型文件...")
    for file in required_files:
        if (model_dir / file).exists():
            size = (model_dir / file).stat().st_size / (1024*1024)
            print(f"✅ {file}: {size:.1f}MB")
        else:
            print(f"❌ {file}: 不存在")
            return
    
    # 加载模型
    print("\n🔄 加载模型组件...")
    try:
        model = joblib.load(model_dir / "ensemble_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        feature_selector = joblib.load(model_dir / "feature_selector.pkl")
        label_encoder = joblib.load(model_dir / "label_encoder.pkl")
        
        print("✅ 模型加载成功！")
        print(f"📊 支持的情感类别: {list(label_encoder.classes_)}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 8种中文情感配置
    emotions_config = {
        'happy': {'chinese': '快乐', 'icon': '😊', 'color': '#FFD700'},
        'calm': {'chinese': '平静', 'icon': '🍃', 'color': '#90EE90'},
        'energetic': {'chinese': '激昂', 'icon': '⚡', 'color': '#FF6347'},
        'melancholic': {'chinese': '忧郁', 'icon': '☁️', 'color': '#9370DB'},
        'nostalgic': {'chinese': '怀念', 'icon': '⏰', 'color': '#CD853F'},
        'romantic': {'chinese': '浪漫', 'icon': '💕', 'color': '#FF69B4'},
        'mysterious': {'chinese': '深沉', 'icon': '🌙', 'color': '#4B0082'},
        'dramatic': {'chinese': '激烈', 'icon': '🎭', 'color': '#DC143C'}
    }
    
    # 使用现有的特征数据进行测试
    print("\n🧪 使用现有特征数据进行测试...")
    
    try:
        # 加载特征数据
        features = np.load('data/features.npy')
        labels_df = pd.read_csv('data/processed/emotion_labels.csv')
        
        print(f"📊 特征数据形状: {features.shape}")
        print(f"📋 标签数据: {len(labels_df)} 个样本")
        
        # 如果特征是3维的，重塑为2维
        if len(features.shape) == 3:
            features = features.reshape(features.shape[0], -1)
            print(f"🔄 特征重塑为: {features.shape}")
        
        # 取一小部分数据进行测试
        test_size = min(20, len(features))
        test_indices = np.random.choice(len(features), test_size, replace=False)
        
        X_test = features[test_indices]
        y_test = labels_df.iloc[test_indices]['emotion'].values
        
        print(f"\n🎯 测试 {test_size} 个样本...")
        
        # 预处理数据
        X_test_selected = feature_selector.transform(X_test)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # 预测
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)
        
        # 转换预测结果
        predicted_emotions = label_encoder.inverse_transform(predictions)
        
        # 显示结果
        print("\n📈 预测结果:")
        print(f"{'序号':<4} {'真实情感':<12} {'预测情感':<12} {'置信度':<8} {'中文情感':<8}")
        print("-" * 60)
        
        correct_predictions = 0
        results = []
        
        for i in range(test_size):
            true_emotion = y_test[i]
            pred_emotion = predicted_emotions[i]
            confidence = np.max(probabilities[i])
            
            # 转换为中文
            chinese_emotion = emotions_config.get(pred_emotion, {}).get('chinese', pred_emotion)
            icon = emotions_config.get(pred_emotion, {}).get('icon', '❓')
            
            is_correct = true_emotion == pred_emotion
            if is_correct:
                correct_predictions += 1
            
            status = "✅" if is_correct else "❌"
            
            print(f"{i+1:<4} {true_emotion:<12} {pred_emotion:<12} {confidence:<8.3f} {chinese_emotion}{icon}")
            
            results.append({
                'true_emotion': true_emotion,
                'predicted_emotion': pred_emotion,
                'chinese_emotion': chinese_emotion,
                'confidence': confidence,
                'correct': is_correct
            })
        
        # 计算准确率
        accuracy = correct_predictions / test_size
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\n📊 测试总结:")
        print(f"   准确率: {accuracy:.3f} ({correct_predictions}/{test_size})")
        print(f"   平均置信度: {avg_confidence:.3f}")
        
        # 统计各情感的预测情况
        print(f"\n🎭 情感预测统计:")
        emotion_stats = {}
        for result in results:
            emotion = result['chinese_emotion']
            if emotion not in emotion_stats:
                emotion_stats[emotion] = {'count': 0, 'correct': 0}
            emotion_stats[emotion]['count'] += 1
            if result['correct']:
                emotion_stats[emotion]['correct'] += 1
        
        for emotion, stats in emotion_stats.items():
            accuracy_rate = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
            icon = next((config['icon'] for config in emotions_config.values() 
                        if config['chinese'] == emotion), '❓')
            print(f"   {emotion}{icon}: {stats['correct']}/{stats['count']} ({accuracy_rate:.3f})")
        
        # 可视化结果
        create_visualization(results, emotions_config)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def create_visualization(results, emotions_config):
    """创建可视化图表"""
    try:
        # 创建结果图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 情感分布
        emotions = [r['chinese_emotion'] for r in results]
        emotion_counts = pd.Series(emotions).value_counts()
        
        # 获取对应的颜色
        colors = []
        for emotion in emotion_counts.index:
            color = '#808080'  # 默认颜色
            for config in emotions_config.values():
                if config['chinese'] == emotion:
                    color = config['color']
                    break
            colors.append(color)
        
        ax1.pie(emotion_counts.values, labels=emotion_counts.index, 
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('预测情感分布', fontsize=14)
        
        # 置信度分布
        confidences = [r['confidence'] for r in results]
        ax2.hist(confidences, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('置信度')
        ax2.set_ylabel('样本数量')
        ax2.set_title('预测置信度分布', fontsize=14)
        ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'平均置信度: {np.mean(confidences):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('8emotions_simple_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 结果图表已保存: 8emotions_simple_test_results.png")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

if __name__ == "__main__":
    test_8emotions_model() 