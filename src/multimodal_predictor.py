#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态音乐情感预测器
融合音频特征和歌词文本的情感分析
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Any
import json

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from lyrics_emotion_analyzer import ChineseLyricsEmotionAnalyzer, MultiModalEmotionFusion

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalMusicEmotionPredictor:
    """多模态音乐情感预测器"""
    
    def __init__(self):
        # 导入并初始化音频预测器
        try:
            from web_app import MusicEmotionPredictor
            self.audio_predictor = MusicEmotionPredictor()
        except ImportError:
            # 备用方案：使用动态预测器
            logger.warning("无法导入MusicEmotionPredictor，使用备用方案")
            self.audio_predictor = None
        
        # 初始化歌词分析器
        self.lyrics_analyzer = ChineseLyricsEmotionAnalyzer()
        
        # 初始化融合器
        self.fusion_engine = MultiModalEmotionFusion()
        
        logger.info("多模态音乐情感预测器初始化完成")
    
    def predict_audio_only(self, audio_path: str) -> Dict[str, Any]:
        """仅基于音频的情感预测"""
        try:
            if self.audio_predictor is None:
                logger.error("音频预测器未初始化")
                return self._get_default_result('audio_only')
            
            result = self.audio_predictor.predict_emotion(audio_path)
            if result:
                result['modality'] = 'audio_only'
                return result
            else:
                return self._get_default_result('audio_only')
        except Exception as e:
            logger.error(f"音频预测失败: {e}")
            return self._get_default_result('audio_only')
    
    def predict_lyrics_only(self, lyrics: str) -> Dict[str, Any]:
        """仅基于歌词的情感预测"""
        try:
            result = self.lyrics_analyzer.analyze_emotion(lyrics)
            result['modality'] = 'lyrics_only'
            return result
        except Exception as e:
            logger.error(f"歌词分析失败: {e}")
            return self._get_default_result('lyrics_only')
    
    def predict_multimodal(self, audio_path: str, lyrics: str) -> Dict[str, Any]:
        """多模态情感预测（音频+歌词）"""
        try:
            # 获取音频分析结果
            audio_result = self.predict_audio_only(audio_path)
            
            # 获取歌词分析结果
            lyrics_result = self.predict_lyrics_only(lyrics)
            
            # 融合两种结果
            fused_result = self.fusion_engine.fuse_emotions(audio_result, lyrics_result)
            fused_result['modality'] = 'multimodal'
            
            return fused_result
            
        except Exception as e:
            logger.error(f"多模态预测失败: {e}")
            return self._get_default_result('multimodal')
    
    def predict_adaptive(self, audio_path: str, lyrics: Optional[str] = None) -> Dict[str, Any]:
        """自适应预测（根据可用信息选择最佳预测方法）"""
        if not lyrics or lyrics.strip() == "":
            # 仅音频预测
            result = self.predict_audio_only(audio_path)
            result['prediction_strategy'] = 'audio_only_fallback'
            return result
        
        elif not audio_path or not os.path.exists(audio_path):
            # 仅歌词预测
            result = self.predict_lyrics_only(lyrics)
            result['prediction_strategy'] = 'lyrics_only_fallback'
            return result
        
        else:
            # 多模态预测
            result = self.predict_multimodal(audio_path, lyrics)
            result['prediction_strategy'] = 'multimodal_optimal'
            return result
    
    def _get_default_result(self, modality: str) -> Dict[str, Any]:
        """获取默认结果"""
        return {
            'predicted_emotion': '平静',
            'confidence': 0.0,
            'all_probabilities': {
                '平静': 0.125, '忧郁': 0.125, '快乐': 0.125, '激昂': 0.125,
                '浪漫': 0.125, '深沉': 0.125, '怀念': 0.125, '激烈': 0.125
            },
            'modality': modality,
            'error': '预测失败，返回默认结果'
        }
    
    def get_prediction_explanation(self, result: Dict[str, Any]) -> str:
        """获取预测结果的解释"""
        modality = result.get('modality', 'unknown')
        predicted_emotion = result.get('predicted_emotion', '未知')
        confidence = result.get('confidence', 0.0)
        
        explanations = {
            'audio_only': f"基于音频分析：预测情感为'{predicted_emotion}'（置信度：{confidence:.1%}）",
            'lyrics_only': f"基于歌词分析：预测情感为'{predicted_emotion}'（置信度：{confidence:.1%}）",
            'multimodal': f"多模态融合分析：预测情感为'{predicted_emotion}'（置信度：{confidence:.1%}）",
            'multimodal_optimal': f"智能多模态分析：预测情感为'{predicted_emotion}'（置信度：{confidence:.1%}）"
        }
        
        base_explanation = explanations.get(modality, f"预测情感为'{predicted_emotion}'")
        
        # 添加详细信息
        if modality == 'multimodal' and 'fusion_weights' in result:
            weights = result['fusion_weights']
            base_explanation += f"\n- 音频权重：{weights.get('audio', 0):.1%}"
            base_explanation += f"\n- 歌词权重：{weights.get('lyrics', 0):.1%}"
        
        if 'lyrics_result' in result and result['lyrics_result'].get('emotion_words_found'):
            emotion_words = [word['word'] for word in result['lyrics_result']['emotion_words_found'][:5]]
            if emotion_words:
                base_explanation += f"\n- 关键情感词：{', '.join(emotion_words)}"
        
        return base_explanation


class TimeSeriesEmotionAnalyzer:
    """时间序列情感分析器"""
    
    def __init__(self):
        self.multimodal_predictor = MultiModalMusicEmotionPredictor()
        
    def analyze_music_collection(self, music_data: list) -> Dict[str, Any]:
        """分析音乐集合的情感分布"""
        results = []
        
        for item in music_data:
            audio_path = item.get('audio_path')
            lyrics = item.get('lyrics', '')
            year = item.get('year')
            title = item.get('title', 'Unknown')
            artist = item.get('artist', 'Unknown')
            
            # 预测情感
            result = self.multimodal_predictor.predict_adaptive(audio_path, lyrics)
            
            # 添加元数据
            result.update({
                'title': title,
                'artist': artist,
                'year': year,
                'audio_path': audio_path,
                'lyrics': lyrics
            })
            
            results.append(result)
        
        return self._analyze_temporal_trends(results)
    
    def _analyze_temporal_trends(self, results: list) -> Dict[str, Any]:
        """分析时间趋势"""
        # 按年份分组
        yearly_emotions = {}
        for result in results:
            year = result.get('year')
            if year:
                if year not in yearly_emotions:
                    yearly_emotions[year] = []
                yearly_emotions[year].append(result['predicted_emotion'])
        
        # 计算每年的情感分布
        yearly_distribution = {}
        for year, emotions in yearly_emotions.items():
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # 计算百分比
            total = len(emotions)
            yearly_distribution[year] = {
                emotion: count / total for emotion, count in emotion_counts.items()
            }
        
        return {
            'total_songs': len(results),
            'year_range': [min(yearly_emotions.keys()), max(yearly_emotions.keys())] if yearly_emotions else [],
            'yearly_distribution': yearly_distribution,
            'overall_trends': self._calculate_trends(yearly_distribution),
            'detailed_results': results
        }
    
    def _calculate_trends(self, yearly_distribution: Dict) -> Dict[str, Any]:
        """计算情感趋势"""
        if not yearly_distribution:
            return {}
        
        # 计算情感变化趋势
        years = sorted(yearly_distribution.keys())
        trends = {}
        
        emotions = ['快乐', '忧郁', '平静', '激昂', '浪漫', '深沉', '怀念', '激烈']
        
        for emotion in emotions:
            values = []
            for year in years:
                value = yearly_distribution[year].get(emotion, 0)
                values.append(value)
            
            if len(values) > 1:
                # 简单的线性趋势计算
                trend = (values[-1] - values[0]) / len(values)
                trends[emotion] = {
                    'trend': 'increasing' if trend > 0.01 else 'decreasing' if trend < -0.01 else 'stable',
                    'change_rate': trend,
                    'initial_value': values[0],
                    'final_value': values[-1]
                }
        
        return trends


def test_multimodal_predictor():
    """测试多模态预测器"""
    predictor = MultiModalMusicEmotionPredictor()
    
    # 测试歌词分析
    test_lyrics = "阳光明媚的日子里，我们一起快乐地唱歌跳舞，幸福的感觉真美好"
    lyrics_result = predictor.predict_lyrics_only(test_lyrics)
    
    print("🎵 多模态预测器测试结果:")
    print(f"歌词分析：{lyrics_result['predicted_emotion']} (置信度：{lyrics_result['confidence']:.1%})")
    print(f"解释：{predictor.get_prediction_explanation(lyrics_result)}")
    
    # 测试音频分析（如果有音频文件）
    audio_files = list(Path("uploads").glob("*.mp3"))
    if audio_files:
        audio_path = str(audio_files[0])
        audio_result = predictor.predict_audio_only(audio_path)
        print(f"\n音频分析：{audio_result['predicted_emotion']} (置信度：{audio_result.get('confidence', 0):.1%})")
        
        # 测试多模态融合
        multimodal_result = predictor.predict_multimodal(audio_path, test_lyrics)
        print(f"多模态融合：{multimodal_result['predicted_emotion']} (置信度：{multimodal_result['confidence']:.1%})")
        print(f"解释：{predictor.get_prediction_explanation(multimodal_result)}")


if __name__ == "__main__":
    test_multimodal_predictor() 