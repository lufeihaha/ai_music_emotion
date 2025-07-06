#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文歌词情感分析模块
支持多种中文情感分析方法和情感词典
"""

import re
import jieba
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
from pathlib import Path
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseLyricsEmotionAnalyzer:
    """中文歌词情感分析器"""
    
    def __init__(self):
        self.emotion_lexicon = {}
        self.emotion_mapping = {
            'positive': ['快乐', '浪漫', '激昂'],
            'negative': ['忧郁', '深沉'],
            'neutral': ['平静', '怀念'],
            'energetic': ['激烈', '激昂']
        }
        
        # 中文情感词典
        self.emotion_words = {
            '快乐': ['快乐', '开心', '高兴', '愉快', '欢乐', '喜悦', '兴奋', '满足', '幸福', '甜蜜'],
            '忧郁': ['忧郁', '悲伤', '难过', '痛苦', '伤心', '哀伤', '忧愁', '沮丧', '失落', '绝望'],
            '平静': ['平静', '宁静', '安静', '淡然', '从容', '安详', '冷静', '理性', '稳定', '和谐'],
            '激昂': ['激昂', '激动', '热血', '澎湃', '豪迈', '雄壮', '壮烈', '昂扬', '奋进', '斗志'],
            '浪漫': ['浪漫', '温柔', '甜蜜', '浪漫', '柔情', '缠绵', '情深', '钟情', '眷恋', '思念'],
            '深沉': ['深沉', '沉重', '深邃', '凝重', '庄重', '肃穆', '严肃', '厚重', '深刻', '内敛'],
            '怀念': ['怀念', '回忆', '往昔', '追忆', '缅怀', '思念', '眷恋', '留恋', '不舍', '惋惜'],
            '激烈': ['激烈', '狂野', '暴躁', '愤怒', '疯狂', '热烈', '火热', '炽热', '燃烧', '爆发']
        }
        
        self.initialize_emotion_lexicon()
    
    def initialize_emotion_lexicon(self):
        """初始化情感词典"""
        self.emotion_lexicon = {}
        for emotion, words in self.emotion_words.items():
            for word in words:
                self.emotion_lexicon[word] = emotion
    
    def preprocess_lyrics(self, lyrics: str) -> List[str]:
        """预处理歌词文本"""
        if not lyrics:
            return []
        
        # 清理文本
        lyrics = re.sub(r'[^\u4e00-\u9fff\w\s]', '', lyrics)  # 保留中文、英文和数字
        lyrics = re.sub(r'\s+', ' ', lyrics)  # 合并多个空格
        
        # 分词
        words = jieba.lcut(lyrics)
        
        # 过滤停用词和短词
        stop_words = {'的', '了', '在', '是', '我', '你', '他', '她', '它', '和', '与', '及', '或', '但', '就', '都', '也', '还', '只', '又', '再', '可', '能', '会', '要', '不', '没', '有', '这', '那', '什么', '怎么', '为什么', '哪里', '什么时候'}
        words = [word for word in words if len(word) > 1 and word not in stop_words]
        
        return words
    
    def analyze_emotion_words(self, words: List[str]) -> Dict[str, float]:
        """分析情感词汇"""
        emotion_counts = Counter()
        total_emotion_words = 0
        
        for word in words:
            if word in self.emotion_lexicon:
                emotion = self.emotion_lexicon[word]
                emotion_counts[emotion] += 1
                total_emotion_words += 1
        
        # 计算情感比例
        if total_emotion_words == 0:
            return {emotion: 0.0 for emotion in self.emotion_words.keys()}
        
        emotion_scores = {}
        for emotion in self.emotion_words.keys():
            emotion_scores[emotion] = emotion_counts[emotion] / total_emotion_words
        
        return emotion_scores
    
    def analyze_context_emotion(self, lyrics: str) -> Dict[str, float]:
        """基于上下文的情感分析"""
        # 简单的上下文分析
        context_patterns = {
            '快乐': [r'笑', r'开心', r'快乐', r'幸福', r'甜蜜'],
            '忧郁': [r'泪', r'哭', r'伤心', r'难过', r'痛苦'],
            '平静': [r'静', r'安', r'宁', r'淡', r'从容'],
            '激昂': [r'燃', r'热血', r'激动', r'澎湃', r'豪迈'],
            '浪漫': [r'爱', r'心', r'情', r'恋', r'思念'],
            '深沉': [r'深', r'沉', r'重', r'严肃', r'庄重'],
            '怀念': [r'回忆', r'往昔', r'曾经', r'过去', r'怀念'],
            '激烈': [r'狂', r'暴', r'愤怒', r'疯狂', r'爆发']
        }
        
        context_scores = {}
        for emotion, patterns in context_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, lyrics))
                score += matches
            context_scores[emotion] = score
        
        # 归一化
        total_score = sum(context_scores.values())
        if total_score > 0:
            context_scores = {k: v / total_score for k, v in context_scores.items()}
        else:
            context_scores = {k: 0.0 for k in context_scores.keys()}
        
        return context_scores
    
    def analyze_emotion(self, lyrics: str) -> Dict[str, any]:
        """综合情感分析"""
        if not lyrics:
            return {
                'predicted_emotion': '平静',
                'confidence': 0.0,
                'all_probabilities': {emotion: 0.0 for emotion in self.emotion_words.keys()},
                'analysis_method': 'lyrics_analysis',
                'word_count': 0,
                'emotion_words_found': []
            }
        
        # 预处理
        words = self.preprocess_lyrics(lyrics)
        
        # 分析情感词汇
        word_emotions = self.analyze_emotion_words(words)
        
        # 分析上下文情感
        context_emotions = self.analyze_context_emotion(lyrics)
        
        # 融合两种分析结果
        final_emotions = {}
        for emotion in self.emotion_words.keys():
            final_emotions[emotion] = 0.6 * word_emotions[emotion] + 0.4 * context_emotions[emotion]
        
        # 找出主要情感
        if all(score == 0 for score in final_emotions.values()):
            predicted_emotion = '平静'
            confidence = 0.0
        else:
            predicted_emotion = max(final_emotions, key=final_emotions.get)
            confidence = final_emotions[predicted_emotion]
        
        # 找出识别到的情感词汇
        emotion_words_found = []
        for word in words:
            if word in self.emotion_lexicon:
                emotion_words_found.append({
                    'word': word,
                    'emotion': self.emotion_lexicon[word]
                })
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'all_probabilities': final_emotions,
            'analysis_method': 'lyrics_analysis',
            'word_count': len(words),
            'emotion_words_found': emotion_words_found[:10]  # 只显示前10个
        }
    
    def analyze_lyrics_file(self, file_path: str) -> Dict[str, any]:
        """分析歌词文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lyrics = f.read()
            return self.analyze_emotion(lyrics)
        except Exception as e:
            logger.error(f"读取歌词文件失败: {e}")
            return self.analyze_emotion("")


class MultiModalEmotionFusion:
    """多模态情感融合器"""
    
    def __init__(self):
        self.fusion_weights = {
            'audio': 0.7,      # 音频权重
            'lyrics': 0.3      # 歌词权重
        }
        
        # 情感映射矩阵（用于处理音频和歌词情感不一致的情况）
        self.emotion_compatibility = {
            '快乐': {'快乐': 1.0, '浪漫': 0.8, '激昂': 0.6, '平静': 0.4, '怀念': 0.3, '深沉': 0.2, '忧郁': 0.1, '激烈': 0.5},
            '忧郁': {'忧郁': 1.0, '深沉': 0.8, '怀念': 0.6, '平静': 0.4, '浪漫': 0.3, '快乐': 0.1, '激昂': 0.2, '激烈': 0.3},
            '平静': {'平静': 1.0, '怀念': 0.7, '浪漫': 0.6, '深沉': 0.5, '快乐': 0.4, '忧郁': 0.4, '激昂': 0.3, '激烈': 0.1},
            '激昂': {'激昂': 1.0, '激烈': 0.8, '快乐': 0.6, '平静': 0.3, '浪漫': 0.4, '深沉': 0.5, '怀念': 0.3, '忧郁': 0.2},
            '浪漫': {'浪漫': 1.0, '快乐': 0.7, '怀念': 0.6, '平静': 0.5, '深沉': 0.4, '忧郁': 0.4, '激昂': 0.3, '激烈': 0.2},
            '深沉': {'深沉': 1.0, '忧郁': 0.8, '怀念': 0.6, '平静': 0.5, '浪漫': 0.4, '激昂': 0.4, '快乐': 0.2, '激烈': 0.3},
            '怀念': {'怀念': 1.0, '深沉': 0.7, '忧郁': 0.6, '平静': 0.6, '浪漫': 0.5, '快乐': 0.3, '激昂': 0.2, '激烈': 0.1},
            '激烈': {'激烈': 1.0, '激昂': 0.8, '深沉': 0.4, '忧郁': 0.3, '快乐': 0.4, '平静': 0.1, '浪漫': 0.2, '怀念': 0.1}
        }
    
    def fuse_emotions(self, audio_result: Dict, lyrics_result: Dict) -> Dict[str, any]:
        """融合音频和歌词的情感分析结果"""
        # 获取音频和歌词的情感概率
        audio_probs = audio_result.get('all_probabilities', {})
        lyrics_probs = lyrics_result.get('all_probabilities', {})
        
        # 确保两个结果有相同的情感类别
        all_emotions = set(audio_probs.keys()) | set(lyrics_probs.keys())
        
        # 标准化概率
        audio_probs = {emotion: audio_probs.get(emotion, 0.0) for emotion in all_emotions}
        lyrics_probs = {emotion: lyrics_probs.get(emotion, 0.0) for emotion in all_emotions}
        
        # 融合概率
        fused_probs = {}
        for emotion in all_emotions:
            audio_score = audio_probs[emotion]
            lyrics_score = lyrics_probs[emotion]
            
            # 考虑情感兼容性
            compatibility_bonus = 0.0
            if audio_result.get('predicted_emotion') and lyrics_result.get('predicted_emotion'):
                audio_emotion = audio_result['predicted_emotion']
                lyrics_emotion = lyrics_result['predicted_emotion']
                if audio_emotion in self.emotion_compatibility and lyrics_emotion in self.emotion_compatibility[audio_emotion]:
                    compatibility_bonus = self.emotion_compatibility[audio_emotion][lyrics_emotion] * 0.1
            
            # 加权融合
            fused_score = (
                self.fusion_weights['audio'] * audio_score +
                self.fusion_weights['lyrics'] * lyrics_score +
                compatibility_bonus
            )
            fused_probs[emotion] = fused_score
        
        # 归一化
        total_prob = sum(fused_probs.values())
        if total_prob > 0:
            fused_probs = {k: v / total_prob for k, v in fused_probs.items()}
        
        # 确定最终情感
        predicted_emotion = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[predicted_emotion]
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'all_probabilities': fused_probs,
            'analysis_method': 'multimodal_fusion',
            'audio_result': audio_result,
            'lyrics_result': lyrics_result,
            'fusion_weights': self.fusion_weights
        }


def test_lyrics_analyzer():
    """测试歌词分析器"""
    analyzer = ChineseLyricsEmotionAnalyzer()
    
    # 测试不同情感的歌词
    test_lyrics = [
        "阳光明媚的日子里，我们一起快乐地唱歌跳舞，幸福的感觉真美好",  # 快乐
        "雨夜里独自一人，泪水模糊了双眼，心中满是忧伤和痛苦",  # 忧郁
        "月光如水，夜晚宁静，心如止水般平静安详",  # 平静
        "热血沸腾的青春，我们要勇敢地去追求梦想，永不放弃",  # 激昂
        "温柔的月光下，我们手牵手走过浪漫的小径，爱意绵绵",  # 浪漫
    ]
    
    print("🎵 歌词情感分析测试结果:")
    for i, lyrics in enumerate(test_lyrics, 1):
        result = analyzer.analyze_emotion(lyrics)
        print(f"\n测试 {i}:")
        print(f"歌词: {lyrics[:30]}...")
        print(f"预测情感: {result['predicted_emotion']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"情感词汇: {[word['word'] for word in result['emotion_words_found']]}")


if __name__ == "__main__":
    test_lyrics_analyzer() 