#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
from collections import Counter

def analyze_user_feedback():
    """分析用户反馈数据"""
    print("📊 用户反馈分析报告")
    print("=" * 50)
    
    # 读取反馈数据
    feedback_data = []
    try:
        with open('data/user_feedback.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    feedback_data.append(data)
    except Exception as e:
        print(f"❌ 读取反馈数据失败: {e}")
        return
    
    print(f"📈 总反馈数量: {len(feedback_data)}")
    
    # 分析预测准确性
    correct_predictions = 0
    total_predictions = 0
    prediction_errors = []
    
    for feedback in feedback_data:
        predicted = feedback.get('predicted_emotion', '')
        actual = feedback.get('actual_emotion', '')
        confidence = feedback.get('confidence', 0)
        rating = feedback.get('user_rating', 0)
        
        if predicted and actual:
            total_predictions += 1
            if predicted == actual:
                correct_predictions += 1
            else:
                prediction_errors.append({
                    'predicted': predicted,
                    'actual': actual,
                    'confidence': confidence,
                    'file': feedback.get('file_name', 'unknown')
                })
    
    # 计算准确率
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"🎯 预测准确率: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    else:
        print("⚠️ 没有可分析的预测数据")
        return
    
    # 分析置信度
    confidences = [f.get('confidence', 0) for f in feedback_data if f.get('confidence')]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"🔮 平均置信度: {avg_confidence:.3f}")
        
        low_confidence_count = len([c for c in confidences if c < 0.5])
        print(f"📉 低置信度预测 (<50%): {low_confidence_count}/{len(confidences)}")
    
    # 分析预测分布
    predicted_emotions = [f.get('predicted_emotion') for f in feedback_data if f.get('predicted_emotion')]
    actual_emotions = [f.get('actual_emotion') for f in feedback_data if f.get('actual_emotion')]
    
    print(f"\n📊 预测情感分布:")
    predicted_counts = Counter(predicted_emotions)
    for emotion, count in predicted_counts.most_common():
        print(f"  {emotion}: {count}次 ({count/len(predicted_emotions):.1%})")
    
    print(f"\n📋 实际情感分布:")
    actual_counts = Counter(actual_emotions)
    for emotion, count in actual_counts.most_common():
        print(f"  {emotion}: {count}次 ({count/len(actual_emotions):.1%})")
    
    # 分析常见错误
    if prediction_errors:
        print(f"\n❌ 预测错误分析 ({len(prediction_errors)}个错误):")
        error_patterns = Counter()
        for error in prediction_errors:
            pattern = f"{error['predicted']} → {error['actual']}"
            error_patterns[pattern] += 1
        
        for pattern, count in error_patterns.most_common(5):
            print(f"  {pattern}: {count}次")
    
    # 分析用户评分
    ratings = [f.get('user_rating', 0) for f in feedback_data if f.get('user_rating', 0) > 0]
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        print(f"\n⭐ 用户评分:")
        print(f"  平均评分: {avg_rating:.1f}/5")
        print(f"  评分数量: {len(ratings)}")
        
        rating_dist = Counter(ratings)
        for rating in sorted(rating_dist.keys()):
            print(f"  {rating}星: {rating_dist[rating]}次")
    
    # 生成改进建议
    print(f"\n💡 改进建议:")
    
    # 1. 准确率问题
    if accuracy < 0.6:
        print(f"  1. 🎯 预测准确率偏低 ({accuracy:.1%})，建议:")
        print(f"     - 增加训练数据，特别是错误预测的情感类型")
        print(f"     - 考虑使用集成学习方法")
        print(f"     - 调整模型超参数")
    
    # 2. 置信度问题
    if confidences and avg_confidence < 0.6:
        print(f"  2. 🔮 预测置信度偏低 ({avg_confidence:.3f})，建议:")
        print(f"     - 设置置信度阈值，低于阈值时提示用户")
        print(f"     - 优化特征工程")
        print(f"     - 使用模型校准技术")
    
    # 3. 数据不平衡问题
    if predicted_emotions:
        most_predicted = predicted_counts.most_common(1)[0]
        if most_predicted[1] / len(predicted_emotions) > 0.5:
            print(f"  3. ⚖️ 预测偏向'{most_predicted[0]}'情感 ({most_predicted[1]/len(predicted_emotions):.1%})，建议:")
            print(f"     - 平衡训练数据中各情感类别的比例")
            print(f"     - 使用类别权重调整")
            print(f"     - 增加其他情感类型的训练样本")
    
    # 4. 特定错误模式
    if prediction_errors and error_patterns:
        most_common_error = error_patterns.most_common(1)[0]
        print(f"  4. 🔄 常见错误模式 '{most_common_error[0]}' ({most_common_error[1]}次)，建议:")
        print(f"     - 针对这种混淆增加特征区分度")
        print(f"     - 收集更多相关标注数据")
        print(f"     - 分析这两种情感的音频特征差异")
    
    # 5. 用户满意度
    if ratings and avg_rating < 3.5:
        print(f"  5. 😞 用户满意度较低 ({avg_rating:.1f}/5)，建议:")
        print(f"     - 优先解决准确率和置信度问题")
        print(f"     - 改进用户界面和交互体验")
        print(f"     - 提供更详细的分析解释")
    
    print(f"\n✅ 分析完成!")
    print(f"📁 建议查看详细的用户反馈文件: data/user_feedback.jsonl")

if __name__ == "__main__":
    analyze_user_feedback() 