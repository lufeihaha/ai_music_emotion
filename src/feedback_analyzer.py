#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """用户反馈数据分析器"""
    
    def __init__(self):
        self.feedback_data = []
        self.analysis_results = {}
        
    def load_feedback_data(self, feedback_file='data/user_feedback.jsonl'):
        """加载用户反馈数据"""
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.feedback_data.append(data)
            
            logger.info(f"加载了 {len(self.feedback_data)} 条用户反馈")
            return True
            
        except Exception as e:
            logger.error(f"加载反馈数据失败: {e}")
            return False
    
    def analyze_prediction_accuracy(self):
        """分析预测准确性"""
        if not self.feedback_data:
            return None
        
        df = pd.DataFrame(self.feedback_data)
        
        # 计算整体准确率
        correct_predictions = df[df['predicted_emotion'] == df['actual_emotion']]
        overall_accuracy = len(correct_predictions) / len(df)
        
        # 按情感类别分析
        emotion_accuracy = {}
        for emotion in df['predicted_emotion'].unique():
            emotion_data = df[df['predicted_emotion'] == emotion]
            if len(emotion_data) > 0:
                correct = len(emotion_data[emotion_data['predicted_emotion'] == emotion_data['actual_emotion']])
                emotion_accuracy[emotion] = {
                    'accuracy': correct / len(emotion_data),
                    'total_predictions': len(emotion_data),
                    'correct_predictions': correct
                }
        
        # 分析常见错误
        mistakes = df[df['predicted_emotion'] != df['actual_emotion']]
        common_mistakes = []
        if len(mistakes) > 0:
            mistake_counts = mistakes.groupby(['predicted_emotion', 'actual_emotion']).size()
            common_mistakes = [(pred, actual, count) for (pred, actual), count in mistake_counts.items()]
            common_mistakes.sort(key=lambda x: x[2], reverse=True)
        
        analysis = {
            'overall_accuracy': overall_accuracy,
            'total_feedback': len(df),
            'emotion_accuracy': emotion_accuracy,
            'common_mistakes': common_mistakes,
            'average_confidence': df['confidence'].mean(),
            'low_confidence_threshold': 0.5,
            'low_confidence_count': len(df[df['confidence'] < 0.5])
        }
        
        self.analysis_results['accuracy'] = analysis
        return analysis
    
    def analyze_confidence_issues(self):
        """分析置信度问题"""
        if not self.feedback_data:
            return None
        
        df = pd.DataFrame(self.feedback_data)
        
        # 置信度分布分析
        confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        confidence_distribution = pd.cut(df['confidence'], bins=confidence_bins).value_counts().sort_index()
        
        # 置信度与准确性关系
        df['is_correct'] = df['predicted_emotion'] == df['actual_emotion']
        confidence_accuracy = df.groupby(pd.cut(df['confidence'], bins=confidence_bins))['is_correct'].mean()
        
        # 低置信度但正确的预测
        low_conf_correct = df[(df['confidence'] < 0.5) & (df['is_correct'] == True)]
        
        # 高置信度但错误的预测
        high_conf_wrong = df[(df['confidence'] > 0.7) & (df['is_correct'] == False)]
        
        analysis = {
            'confidence_distribution': confidence_distribution.to_dict(),
            'confidence_accuracy_relation': confidence_accuracy.to_dict(),
            'low_confidence_correct_count': len(low_conf_correct),
            'high_confidence_wrong_count': len(high_conf_wrong),
            'average_confidence': df['confidence'].mean(),
            'confidence_std': df['confidence'].std()
        }
        
        self.analysis_results['confidence'] = analysis
        return analysis
    
    def analyze_user_satisfaction(self):
        """分析用户满意度"""
        if not self.feedback_data:
            return None
        
        df = pd.DataFrame(self.feedback_data)
        
        # 过滤有效评分
        rated_feedback = df[df['user_rating'] > 0]
        
        if len(rated_feedback) == 0:
            return {'message': '暂无用户评分数据'}
        
        analysis = {
            'average_rating': rated_feedback['user_rating'].mean(),
            'rating_distribution': rated_feedback['user_rating'].value_counts().sort_index().to_dict(),
            'total_ratings': len(rated_feedback),
            'satisfaction_rate': len(rated_feedback[rated_feedback['user_rating'] >= 4]) / len(rated_feedback)
        }
        
        self.analysis_results['satisfaction'] = analysis
        return analysis
    
    def identify_problem_patterns(self):
        """识别问题模式"""
        if not self.feedback_data:
            return None
        
        df = pd.DataFrame(self.feedback_data)
        
        problems = []
        
        # 1. 系统性偏向某种情感
        predicted_counts = df['predicted_emotion'].value_counts()
        if predicted_counts.max() / len(df) > 0.5:
            most_predicted = predicted_counts.index[0]
            problems.append({
                'type': 'bias_towards_emotion',
                'description': f'系统过度倾向于预测"{most_predicted}"情感',
                'severity': 'high',
                'affected_percentage': predicted_counts.max() / len(df),
                'emotion': most_predicted
            })
        
        # 2. 特定情感识别困难
        for emotion in df['actual_emotion'].unique():
            emotion_data = df[df['actual_emotion'] == emotion]
            if len(emotion_data) >= 2:  # 至少有2个样本
                correct_rate = len(emotion_data[emotion_data['predicted_emotion'] == emotion]) / len(emotion_data)
                if correct_rate < 0.3:
                    problems.append({
                        'type': 'poor_emotion_recognition',
                        'description': f'对"{emotion}"情感识别困难',
                        'severity': 'medium',
                        'accuracy': correct_rate,
                        'emotion': emotion,
                        'sample_count': len(emotion_data)
                    })
        
        # 3. 低置信度问题
        low_confidence_rate = len(df[df['confidence'] < 0.5]) / len(df)
        if low_confidence_rate > 0.3:
            problems.append({
                'type': 'low_confidence',
                'description': '预测置信度普遍较低',
                'severity': 'medium',
                'low_confidence_rate': low_confidence_rate,
                'average_confidence': df['confidence'].mean()
            })
        
        self.analysis_results['problems'] = problems
        return problems
    
    def generate_improvement_suggestions(self):
        """生成改进建议"""
        suggestions = []
        
        # 基于准确性分析的建议
        if 'accuracy' in self.analysis_results:
            accuracy_data = self.analysis_results['accuracy']
            
            if accuracy_data['overall_accuracy'] < 0.6:
                suggestions.append({
                    'category': 'model_performance',
                    'priority': 'high',
                    'suggestion': '整体准确率较低，建议重新训练模型或增加训练数据',
                    'current_value': accuracy_data['overall_accuracy'],
                    'target_value': 0.7
                })
            
            # 针对特定情感的建议
            for emotion, data in accuracy_data['emotion_accuracy'].items():
                if data['accuracy'] < 0.5 and data['total_predictions'] >= 2:
                    suggestions.append({
                        'category': 'emotion_specific',
                        'priority': 'medium',
                        'suggestion': f'"{emotion}"情感识别准确率较低，建议增加相关训练样本',
                        'emotion': emotion,
                        'current_accuracy': data['accuracy'],
                        'sample_count': data['total_predictions']
                    })
        
        # 基于置信度分析的建议
        if 'confidence' in self.analysis_results:
            confidence_data = self.analysis_results['confidence']
            
            if confidence_data['average_confidence'] < 0.6:
                suggestions.append({
                    'category': 'confidence',
                    'priority': 'medium',
                    'suggestion': '预测置信度偏低，建议调整模型参数或使用集成学习',
                    'current_confidence': confidence_data['average_confidence'],
                    'target_confidence': 0.7
                })
        
        # 基于问题模式的建议
        if 'problems' in self.analysis_results:
            for problem in self.analysis_results['problems']:
                if problem['type'] == 'bias_towards_emotion':
                    suggestions.append({
                        'category': 'data_balance',
                        'priority': 'high',
                        'suggestion': f'数据不平衡，过度预测"{problem["emotion"]}"，需要平衡训练数据',
                        'biased_emotion': problem['emotion'],
                        'bias_rate': problem['affected_percentage']
                    })
        
        return suggestions
    
    def create_visualizations(self):
        """创建可视化图表"""
        if not self.feedback_data:
            return
        
        df = pd.DataFrame(self.feedback_data)
        
        # 创建保存目录
        os.makedirs('models/feedback_analysis', exist_ok=True)
        
        # 1. 预测准确性热力图
        plt.figure(figsize=(10, 8))
        confusion_data = pd.crosstab(df['actual_emotion'], df['predicted_emotion'], normalize='index')
        sns.heatmap(confusion_data, annot=True, fmt='.2f', cmap='Blues')
        plt.title('情感预测准确性热力图')
        plt.xlabel('预测情感')
        plt.ylabel('实际情感')
        plt.tight_layout()
        plt.savefig('models/feedback_analysis/prediction_accuracy_heatmap.png', dpi=300)
        plt.close()
        
        # 2. 置信度分布图
        plt.figure(figsize=(10, 6))
        plt.hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=df['confidence'].mean(), color='red', linestyle='--', 
                   label=f'平均置信度: {df["confidence"].mean():.3f}')
        plt.xlabel('预测置信度')
        plt.ylabel('频次')
        plt.title('预测置信度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/feedback_analysis/confidence_distribution.png', dpi=300)
        plt.close()
        
        # 3. 情感预测分布
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        df['predicted_emotion'].value_counts().plot(kind='bar', alpha=0.7)
        plt.title('预测情感分布')
        plt.xlabel('情感类别')
        plt.ylabel('预测次数')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        df['actual_emotion'].value_counts().plot(kind='bar', alpha=0.7, color='orange')
        plt.title('实际情感分布')
        plt.xlabel('情感类别')
        plt.ylabel('实际次数')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('models/feedback_analysis/emotion_distribution.png', dpi=300)
        plt.close()
        
        logger.info("可视化图表已保存到 models/feedback_analysis/")
    
    def generate_report(self):
        """生成完整的分析报告"""
        # 进行所有分析
        accuracy_analysis = self.analyze_prediction_accuracy()
        confidence_analysis = self.analyze_confidence_issues()
        satisfaction_analysis = self.analyze_user_satisfaction()
        problems = self.identify_problem_patterns()
        suggestions = self.generate_improvement_suggestions()
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_feedback': len(self.feedback_data),
                'overall_accuracy': accuracy_analysis['overall_accuracy'] if accuracy_analysis else 0,
                'average_confidence': accuracy_analysis['average_confidence'] if accuracy_analysis else 0,
                'average_rating': satisfaction_analysis.get('average_rating', 0) if satisfaction_analysis else 0
            },
            'detailed_analysis': {
                'accuracy': accuracy_analysis,
                'confidence': confidence_analysis,
                'satisfaction': satisfaction_analysis,
                'problems': problems,
                'suggestions': suggestions
            }
        }
        
        # 保存报告
        os.makedirs('models/feedback_analysis', exist_ok=True)
        report_file = f'models/feedback_analysis/feedback_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析报告已保存: {report_file}")
        return report

def main():
    """主函数"""
    print("📊 用户反馈数据分析")
    print("=" * 50)
    
    analyzer = FeedbackAnalyzer()
    
    # 加载数据
    if not analyzer.load_feedback_data():
        print("❌ 无法加载反馈数据")
        return
    
    # 生成报告
    report = analyzer.generate_report()
    
    # 创建可视化
    analyzer.create_visualizations()
    
    # 打印关键结果
    print(f"\n📈 分析结果摘要:")
    print(f"  总反馈数: {report['summary']['total_feedback']}")
    print(f"  整体准确率: {report['summary']['overall_accuracy']:.1%}")
    print(f"  平均置信度: {report['summary']['average_confidence']:.3f}")
    
    if report['summary']['average_rating'] > 0:
        print(f"  平均评分: {report['summary']['average_rating']:.1f}/5")
    
    # 打印主要问题
    if report['detailed_analysis']['problems']:
        print(f"\n⚠️ 发现的主要问题:")
        for i, problem in enumerate(report['detailed_analysis']['problems'][:3], 1):
            print(f"  {i}. {problem['description']}")
    
    # 打印改进建议
    if report['detailed_analysis']['suggestions']:
        print(f"\n💡 改进建议:")
        high_priority = [s for s in report['detailed_analysis']['suggestions'] if s['priority'] == 'high']
        for i, suggestion in enumerate(high_priority[:3], 1):
            print(f"  {i}. {suggestion['suggestion']}")
    
    print(f"\n✅ 分析完成! 详细结果保存在 models/feedback_analysis/")

if __name__ == "__main__":
    main() 