#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成实验报告性能分析章节的图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from datetime import datetime
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_model_performance_comparison():
    """创建模型性能对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 模型准确率对比
    models = ['Random Forest', 'SVM', 'Gradient Boosting', 'MLP', '集成模型', '用户反馈现状']
    accuracies = [61.67, 67.67, 66.00, 66.00, 68.67, 0.0]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#E91E63', '#F44336']
    
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('📊 模型性能对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率 (%)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 75)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # 添加警告线
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='可接受阈值')
    ax1.legend()
    
    # 2. 用户反馈问题分析
    problems = ['预测准确率', '置信度水平', '用户满意度', '响应速度', '界面友好度']
    current_scores = [0, 30.7, 40, 85, 75]  # 0%, 30.7%, 2/5*20=40%, 85%, 75%
    target_scores = [70, 70, 80, 90, 85]
    
    x = np.arange(len(problems))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, current_scores, width, label='当前水平', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, target_scores, width, label='目标水平', color='#4ECDC4', alpha=0.8)
    
    ax2.set_title('📈 性能改进目标', fontsize=14, fontweight='bold')
    ax2.set_ylabel('得分')
    ax2.set_xticks(x)
    ax2.set_xticklabels(problems, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('performance_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_user_feedback_analysis():
    """创建用户反馈分析图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 预测vs实际情感分布
    predicted_emotions = ['平静', '激昂']
    predicted_counts = [3, 2]
    actual_emotions = ['怀念', '浪漫', '忧郁']
    actual_counts = [3, 1, 1]
    
    ax1.pie(predicted_counts, labels=predicted_emotions, autopct='%1.1f%%',
           colors=['#87CEEB', '#FF6347'], startangle=90)
    ax1.set_title('🔮 系统预测情感分布', fontsize=12, fontweight='bold')
    
    ax2.pie(actual_counts, labels=actual_emotions, autopct='%1.1f%%',
           colors=['#DDA0DD', '#FFB6C1', '#B0C4DE'], startangle=90)
    ax2.set_title('🎯 用户实际需求分布', fontsize=12, fontweight='bold')
    
    # 3. 错误模式分析
    error_patterns = ['激昂→怀念', '平静→浪漫', '平静→忧郁', '平静→怀念']
    error_counts = [2, 1, 1, 1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    bars = ax3.bar(error_patterns, error_counts, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title('❌ 常见预测错误模式', fontsize=12, fontweight='bold')
    ax3.set_ylabel('错误次数')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, error_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 置信度分布
    confidence_ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    confidence_counts = [1, 4, 0, 0, 0]  # 基于实际数据：0.307平均值
    
    ax4.bar(confidence_ranges, confidence_counts, color='skyblue', alpha=0.8, edgecolor='black')
    ax4.set_title('📊 预测置信度分布', fontsize=12, fontweight='bold')
    ax4.set_ylabel('预测次数')
    ax4.tick_params(axis='x', rotation=45)
    
    # 添加低置信度警告线
    ax4.axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='问题阈值')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('user_feedback_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_roadmap():
    """创建改进路线图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 定义改进阶段
    stages = [
        ("当前状态", 0.1, 0.2, "准确率: 0%\n置信度: 30.7%\n满意度: 2/5", '#F44336'),
        ("短期改进\n(1个月)", 0.3, 0.4, "准确率: 60%\n置信度: 60%\n满意度: 3.5/5", '#FF9800'),
        ("中期目标\n(3个月)", 0.5, 0.6, "准确率: 75%\n置信度: 75%\n满意度: 4/5", '#2196F3'),
        ("长期愿景\n(6个月)", 0.7, 0.8, "准确率: 85%\n置信度: 85%\n满意度: 4.5/5", '#4CAF50'),
    ]
    
    # 绘制阶段框
    for i, (stage_name, x, y, description, color) in enumerate(stages):
        # 主要阶段框
        rect = patches.Rectangle((x, y), 0.15, 0.3, linewidth=2, 
                               edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        
        # 阶段标题
        ax.text(x + 0.075, y + 0.25, stage_name, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        
        # 描述文字
        ax.text(x + 0.075, y - 0.05, description, ha='center', va='top', 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 添加箭头（除了最后一个）
        if i < len(stages) - 1:
            ax.annotate('', xy=(stages[i+1][1], y + 0.15), 
                       xytext=(x + 0.15, y + 0.15),
                       arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    # 添加改进措施
    improvements = [
        ("数据增强\n特征优化\n模型集成", 0.2, 0.65),
        ("用户反馈\n持续训练\n界面优化", 0.4, 0.65),
        ("深度学习\n多模态融合\n个性化", 0.6, 0.65),
    ]
    
    for text, x, y in improvements:
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('🚀 系统改进路线图', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('improvement_roadmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_detailed():
    """创建详细特征重要性分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 特征类别重要性
    feature_categories = ['MFCC特征', '频谱特征', '色度特征', '节奏特征', '和声特征', '其他特征']
    importance_scores = [0.25, 0.20, 0.30, 0.15, 0.08, 0.02]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#DDA0DD']
    
    bars = ax1.barh(feature_categories, importance_scores, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('🎵 音频特征类别重要性', fontsize=14, fontweight='bold')
    ax1.set_xlabel('重要性得分')
    
    # 添加数值标签
    for bar, score in zip(bars, importance_scores):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', ha='left', va='center', fontweight='bold')
    
    # 2. Top-10 重要特征
    feature_names = ['色度特征40', 'MFCC-1', '频谱中心', '色度特征38', 'MFCC-2', 
                    '零交叉率', 'RMS能量', '色度特征39', 'MFCC-3', '频谱带宽']
    feature_scores = [0.032, 0.029, 0.022, 0.019, 0.018, 0.017, 0.016, 0.014, 0.013, 0.012]
    
    bars = ax2.bar(range(len(feature_names)), feature_scores, 
                   color='skyblue', alpha=0.8, edgecolor='black')
    ax2.set_title('🔝 Top-10 重要特征', fontsize=14, fontweight='bold')
    ax2.set_ylabel('重要性得分')
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    
    # 添加数值标签
    for bar, score in zip(bars, feature_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix_analysis():
    """创建混淆矩阵分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 理想混淆矩阵（目标）
    emotions = ['快乐', '平静', '激昂', '忧郁', '怀念', '浪漫', '深沉', '激烈']
    ideal_matrix = np.eye(8) * 0.8 + np.random.normal(0, 0.05, (8, 8))
    ideal_matrix = np.clip(ideal_matrix, 0, 1)
    np.fill_diagonal(ideal_matrix, 0.85)  # 对角线设为0.85
    
    sns.heatmap(ideal_matrix, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=emotions, yticklabels=emotions, ax=ax1)
    ax1.set_title('🎯 理想混淆矩阵 (目标85%准确率)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('预测情感')
    ax1.set_ylabel('实际情感')
    
    # 2. 当前问题矩阵（基于用户反馈）
    current_matrix = np.zeros((5, 5))
    # 基于用户反馈数据构造
    emotions_current = ['激昂', '平静', '怀念', '浪漫', '忧郁']
    
    # 激昂→怀念: 2次错误
    current_matrix[0, 2] = 0.4  # 激昂预测为怀念
    # 平静→其他: 3次错误
    current_matrix[1, 2] = 0.2  # 平静预测为怀念
    current_matrix[1, 3] = 0.2  # 平静预测为浪漫
    current_matrix[1, 4] = 0.2  # 平静预测为忧郁
    
    sns.heatmap(current_matrix, annot=True, fmt='.2f', cmap='Reds',
               xticklabels=emotions_current, yticklabels=emotions_current, ax=ax2)
    ax2.set_title('❌ 当前问题矩阵 (0%准确率)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('预测情感')
    ax2.set_ylabel('实际情感')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数 - 生成所有性能分析图表"""
    print("📊 开始生成性能分析图表...")
    
    try:
        print("1. 生成模型性能对比图...")
        create_model_performance_comparison()
        
        print("2. 生成用户反馈分析图...")
        create_user_feedback_analysis()
        
        print("3. 生成改进路线图...")
        create_improvement_roadmap()
        
        print("4. 生成特征重要性详细分析...")
        create_feature_importance_detailed()
        
        print("5. 生成混淆矩阵分析...")
        create_confusion_matrix_analysis()
        
        print("✅ 所有性能分析图表生成完成！")
        print("📁 图表保存在当前目录：")
        print("  - performance_analysis_comparison.png")
        print("  - user_feedback_analysis.png")
        print("  - improvement_roadmap.png")
        print("  - feature_importance_detailed.png")
        print("  - confusion_matrix_analysis.png")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")

if __name__ == "__main__":
    main() 