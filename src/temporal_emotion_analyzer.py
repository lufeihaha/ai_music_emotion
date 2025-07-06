#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列音乐情感分析系统
分析中文音乐在不同年代的情感变化趋势
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from multimodal_predictor import MultiModalMusicEmotionPredictor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ChineseMusicTemporalAnalyzer:
    """中文音乐时间序列情感分析器"""
    
    def __init__(self):
        self.predictor = MultiModalMusicEmotionPredictor()
        self.emotions = ['快乐', '忧郁', '平静', '激昂', '浪漫', '深沉', '怀念', '激烈']
        
        # 历史数据存储
        self.music_database = []
        self.analysis_results = {}
        
        logger.info("时间序列分析器初始化完成")
    
    def add_music_data(self, title: str, artist: str, year: int, 
                      audio_path: Optional[str] = None, lyrics: Optional[str] = None,
                      genre: Optional[str] = None, album: Optional[str] = None):
        """添加音乐数据到数据库"""
        music_entry = {
            'title': title,
            'artist': artist,
            'year': year,
            'audio_path': audio_path,
            'lyrics': lyrics,
            'genre': genre,
            'album': album,
            'id': len(self.music_database)
        }
        
        self.music_database.append(music_entry)
        logger.info(f"添加音乐: {artist} - {title} ({year})")
    
    def load_music_dataset(self, dataset_path: str):
        """从文件加载音乐数据集"""
        try:
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                data = df.to_dict('records')
            else:
                raise ValueError("不支持的文件格式，请使用JSON或CSV")
            
            for item in data:
                self.add_music_data(
                    title=item.get('title', ''),
                    artist=item.get('artist', ''),
                    year=int(item.get('year', 0)),
                    audio_path=item.get('audio_path'),
                    lyrics=item.get('lyrics'),
                    genre=item.get('genre'),
                    album=item.get('album')
                )
            
            logger.info(f"成功加载 {len(data)} 首音乐数据")
            
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def analyze_temporal_trends(self, start_year: int = 1990, end_year: int = 2024) -> Dict[str, Any]:
        """分析时间序列情感趋势"""
        logger.info(f"开始分析 {start_year}-{end_year} 年的情感趋势")
        
        # 过滤年份范围内的音乐
        filtered_music = [
            music for music in self.music_database 
            if start_year <= music['year'] <= end_year
        ]
        
        if not filtered_music:
            raise ValueError(f"在 {start_year}-{end_year} 年范围内没有找到音乐数据")
        
        # 分析每首歌的情感
        yearly_emotions = {}
        detailed_results = []
        
        for i, music in enumerate(filtered_music):
            logger.info(f"分析进度: {i+1}/{len(filtered_music)} - {music['artist']} - {music['title']}")
            
            try:
                # 进行情感分析
                result = self.predictor.predict_adaptive(
                    audio_path=music['audio_path'],
                    lyrics=music['lyrics']
                )
                
                # 添加音乐信息
                result.update({
                    'title': music['title'],
                    'artist': music['artist'],
                    'year': music['year'],
                    'genre': music['genre'],
                    'album': music['album']
                })
                
                detailed_results.append(result)
                
                # 按年份分组
                year = music['year']
                if year not in yearly_emotions:
                    yearly_emotions[year] = []
                yearly_emotions[year].append(result['predicted_emotion'])
                
            except Exception as e:
                logger.warning(f"分析失败 {music['title']}: {e}")
                continue
        
        # 计算趋势
        trends = self._calculate_temporal_trends(yearly_emotions)
        
        # 保存结果
        self.analysis_results = {
            'start_year': start_year,
            'end_year': end_year,
            'total_songs': len(detailed_results),
            'yearly_emotions': yearly_emotions,
            'trends': trends,
            'detailed_results': detailed_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"时间序列分析完成，分析了 {len(detailed_results)} 首音乐")
        return self.analysis_results
    
    def _calculate_temporal_trends(self, yearly_emotions: Dict[int, List[str]]) -> Dict[str, Any]:
        """计算时间趋势"""
        years = sorted(yearly_emotions.keys())
        
        # 计算每年每种情感的比例
        yearly_proportions = {}
        for year in years:
            emotions_this_year = yearly_emotions[year]
            total_songs = len(emotions_this_year)
            
            proportions = {}
            for emotion in self.emotions:
                count = emotions_this_year.count(emotion)
                proportions[emotion] = count / total_songs if total_songs > 0 else 0
            
            yearly_proportions[year] = proportions
        
        # 计算趋势（线性回归）
        emotion_trends = {}
        for emotion in self.emotions:
            values = [yearly_proportions[year][emotion] for year in years]
            
            if len(values) > 1:
                # 简单线性趋势计算
                x = np.array(range(len(years)))
                y = np.array(values)
                
                # 计算趋势斜率
                slope = np.polyfit(x, y, 1)[0]
                
                # 计算变化幅度
                change_rate = (values[-1] - values[0]) / len(years) if len(years) > 1 else 0
                
                trend_direction = 'increasing' if slope > 0.005 else 'decreasing' if slope < -0.005 else 'stable'
                
                emotion_trends[emotion] = {
                    'trend_direction': trend_direction,
                    'slope': slope,
                    'change_rate': change_rate,
                    'initial_proportion': values[0],
                    'final_proportion': values[-1],
                    'peak_year': years[np.argmax(values)],
                    'peak_proportion': max(values),
                    'low_year': years[np.argmin(values)],
                    'low_proportion': min(values)
                }
        
        # 识别主要趋势
        major_trends = self._identify_major_trends(emotion_trends)
        
        # 社会事件关联分析
        social_events = self._correlate_with_social_events(yearly_proportions, years)
        
        return {
            'yearly_proportions': yearly_proportions,
            'emotion_trends': emotion_trends,
            'major_trends': major_trends,
            'social_correlations': social_events,
            'summary': self._generate_trend_summary(emotion_trends, years)
        }
    
    def _identify_major_trends(self, emotion_trends: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """识别主要情感趋势"""
        major_trends = []
        
        for emotion, trend_data in emotion_trends.items():
            change_rate = abs(trend_data['change_rate'])
            
            if change_rate > 0.01:  # 显著变化阈值
                major_trends.append({
                    'emotion': emotion,
                    'direction': trend_data['trend_direction'],
                    'magnitude': change_rate,
                    'significance': 'high' if change_rate > 0.02 else 'medium'
                })
        
        # 按变化幅度排序
        major_trends.sort(key=lambda x: x['magnitude'], reverse=True)
        
        return major_trends[:5]  # 返回前5个主要趋势
    
    def _correlate_with_social_events(self, yearly_proportions: Dict, years: List[int]) -> Dict[str, Any]:
        """关联社会事件分析"""
        # 定义一些重要的社会事件年份
        social_events = {
            1997: "香港回归",
            1999: "澳门回归", 
            2001: "加入WTO",
            2003: "SARS疫情",
            2008: "北京奥运会/金融危机",
            2010: "上海世博会",
            2020: "COVID-19疫情",
            2022: "北京冬奥会"
        }
        
        event_correlations = []
        
        for event_year, event_name in social_events.items():
            if event_year in years:
                # 分析事件前后的情感变化
                year_index = years.index(event_year)
                
                # 获取事件前后的情感比例
                if year_index > 0 and year_index < len(years) - 1:
                    before_emotions = yearly_proportions[years[year_index - 1]]
                    event_emotions = yearly_proportions[event_year]
                    after_emotions = yearly_proportions[years[year_index + 1]]
                    
                    # 计算变化
                    changes = {}
                    for emotion in self.emotions:
                        before_to_event = event_emotions[emotion] - before_emotions[emotion]
                        event_to_after = after_emotions[emotion] - event_emotions[emotion]
                        
                        changes[emotion] = {
                            'before_to_event': before_to_event,
                            'event_to_after': event_to_after,
                            'total_impact': before_to_event + event_to_after
                        }
                    
                    # 找出影响最大的情感
                    most_impacted = max(changes.keys(), key=lambda e: abs(changes[e]['total_impact']))
                    
                    event_correlations.append({
                        'year': event_year,
                        'event': event_name,
                        'most_impacted_emotion': most_impacted,
                        'impact_magnitude': abs(changes[most_impacted]['total_impact']),
                        'changes': changes
                    })
        
        return {
            'events_analyzed': len(event_correlations),
            'correlations': event_correlations
        }
    
    def _generate_trend_summary(self, emotion_trends: Dict, years: List[int]) -> str:
        """生成趋势总结"""
        summary_parts = []
        
        # 时间范围
        summary_parts.append(f"📅 分析时间范围: {min(years)}-{max(years)}年")
        
        # 上升趋势的情感
        increasing_emotions = [
            emotion for emotion, data in emotion_trends.items()
            if data['trend_direction'] == 'increasing'
        ]
        if increasing_emotions:
            summary_parts.append(f"📈 上升趋势情感: {', '.join(increasing_emotions)}")
        
        # 下降趋势的情感
        decreasing_emotions = [
            emotion for emotion, data in emotion_trends.items()
            if data['trend_direction'] == 'decreasing'
        ]
        if decreasing_emotions:
            summary_parts.append(f"📉 下降趋势情感: {', '.join(decreasing_emotions)}")
        
        # 稳定情感
        stable_emotions = [
            emotion for emotion, data in emotion_trends.items()
            if data['trend_direction'] == 'stable'
        ]
        if stable_emotions:
            summary_parts.append(f"➡️ 稳定情感: {', '.join(stable_emotions)}")
        
        return '\n'.join(summary_parts)
    
    def create_visualization(self, save_path: Optional[str] = None) -> str:
        """创建时间序列可视化图表"""
        if not self.analysis_results:
            raise ValueError("请先运行时间序列分析")
        
        trends = self.analysis_results['trends']
        yearly_proportions = trends['yearly_proportions']
        years = sorted(yearly_proportions.keys())
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 情感趋势线图
        for emotion in self.emotions:
            values = [yearly_proportions[year][emotion] for year in years]
            ax1.plot(years, values, marker='o', linewidth=2, label=emotion)
        
        ax1.set_title('中文音乐情感变化趋势', fontsize=16, fontweight='bold')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('情感比例')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 情感变化热力图
        emotion_matrix = []
        for emotion in self.emotions:
            values = [yearly_proportions[year][emotion] for year in years]
            emotion_matrix.append(values)
        
        im = ax2.imshow(emotion_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_title('情感分布热力图', fontsize=16, fontweight='bold')
        ax2.set_xticks(range(len(years)))
        ax2.set_xticklabels([str(year) for year in years], rotation=45)
        ax2.set_yticks(range(len(self.emotions)))
        ax2.set_yticklabels(self.emotions)
        
        # 添加数值标签
        for i in range(len(self.emotions)):
            for j in range(len(years)):
                value = emotion_matrix[i][j]
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color='white' if value > 0.5 else 'black', fontsize=8)
        
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # 3. 主要趋势分析
        ax3.axis('off')
        
        major_trends = trends['major_trends']
        trend_text = ["🔍 主要情感趋势分析:", ""]
        
        for i, trend in enumerate(major_trends[:5], 1):
            direction_icon = "📈" if trend['direction'] == 'increasing' else "📉" if trend['direction'] == 'decreasing' else "➡️"
            trend_text.append(f"{i}. {direction_icon} {trend['emotion']}: {trend['direction']}")
            trend_text.append(f"   变化幅度: {trend['magnitude']:.3f} ({trend['significance']})")
            trend_text.append("")
        
        # 添加总结
        trend_text.append("📋 趋势总结:")
        trend_text.extend(trends['summary'].split('\n'))
        
        for i, text in enumerate(trend_text):
            ax3.text(0.05, 0.95 - i*0.05, text, fontsize=11, transform=ax3.transAxes,
                    verticalalignment='top')
        
        ax3.set_title('趋势分析报告', fontsize=16, fontweight='bold')
        
        # 4. 社会事件关联分析
        ax4.axis('off')
        
        social_corr = trends['social_correlations']
        event_text = ["🌟 社会事件关联分析:", ""]
        
        if social_corr['events_analyzed'] > 0:
            for event in social_corr['correlations'][:5]:
                event_text.append(f"📅 {event['year']}: {event['event']}")
                event_text.append(f"   主要影响情感: {event['most_impacted_emotion']}")
                event_text.append(f"   影响强度: {event['impact_magnitude']:.3f}")
                event_text.append("")
        else:
            event_text.append("暂无社会事件关联数据")
        
        for i, text in enumerate(event_text):
            ax4.text(0.05, 0.95 - i*0.08, text, fontsize=11, transform=ax4.transAxes,
                    verticalalignment='top')
        
        ax4.set_title('社会事件影响分析', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'temporal_analysis_{timestamp}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"时间序列可视化图表已保存: {save_path}")
        return save_path
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """生成详细的分析报告"""
        if not self.analysis_results:
            raise ValueError("请先运行时间序列分析")
        
        # 创建报告
        report_sections = []
        
        # 标题
        report_sections.append("# 中文音乐情感时间序列分析报告\n")
        report_sections.append(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        
        # 基本信息
        results = self.analysis_results
        report_sections.append("## 📊 分析概况\n")
        report_sections.append(f"- **分析时间范围**: {results['start_year']}-{results['end_year']}年")
        report_sections.append(f"- **分析歌曲总数**: {results['total_songs']}首")
        report_sections.append(f"- **涵盖年份**: {len(results['yearly_emotions'])}年\n")
        
        # 主要发现
        trends = results['trends']
        report_sections.append("## 🔍 主要发现\n")
        report_sections.append(trends['summary'])
        report_sections.append("")
        
        # 详细趋势分析
        report_sections.append("## 📈 详细趋势分析\n")
        
        for emotion, trend_data in trends['emotion_trends'].items():
            report_sections.append(f"### {emotion}")
            report_sections.append(f"- **总体趋势**: {trend_data['trend_direction']}")
            report_sections.append(f"- **变化率**: {trend_data['change_rate']:.4f}")
            report_sections.append(f"- **初期比例**: {trend_data['initial_proportion']:.2%}")
            report_sections.append(f"- **最终比例**: {trend_data['final_proportion']:.2%}")
            report_sections.append(f"- **峰值年份**: {trend_data['peak_year']} ({trend_data['peak_proportion']:.2%})")
            report_sections.append(f"- **低谷年份**: {trend_data['low_year']} ({trend_data['low_proportion']:.2%})")
            report_sections.append("")
        
        # 社会事件影响
        if trends['social_correlations']['events_analyzed'] > 0:
            report_sections.append("## 🌟 社会事件影响分析\n")
            
            for event in trends['social_correlations']['correlations']:
                report_sections.append(f"### {event['year']}年: {event['event']}")
                report_sections.append(f"- **主要影响情感**: {event['most_impacted_emotion']}")
                report_sections.append(f"- **影响强度**: {event['impact_magnitude']:.3f}")
                report_sections.append("")
        
        # 合并报告
        full_report = '\n'.join(report_sections)
        
        # 保存报告
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'temporal_analysis_report_{timestamp}.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        logger.info(f"分析报告已保存: {output_path}")
        return output_path
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """保存分析结果"""
        if not self.analysis_results:
            raise ValueError("没有分析结果需要保存")
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'temporal_analysis_results_{timestamp}.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存: {output_path}")
        return output_path


def create_sample_dataset():
    """创建示例数据集"""
    sample_data = [
        {
            "title": "千里之外",
            "artist": "周杰伦",
            "year": 2006,
            "lyrics": "千里之外，你无声黑白，沉默年代，或许不该，太遥远的相爱",
            "genre": "流行"
        },
        {
            "title": "青花瓷",
            "artist": "周杰伦", 
            "year": 2007,
            "lyrics": "炊烟袅袅升起，隔江千万里，在瓶底书汉隶仿前朝的飘逸",
            "genre": "中国风"
        },
        {
            "title": "小幸运",
            "artist": "田馥甄",
            "year": 2015,
            "lyrics": "我听见雨滴落在青青草地，我听见远方下课钟声响起",
            "genre": "流行"
        },
        {
            "title": "演员",
            "artist": "薛之谦",
            "year": 2015,
            "lyrics": "简单点，说话的方式简单点，递进的情绪请省略",
            "genre": "流行"
        },
        {
            "title": "成都",
            "artist": "赵雷",
            "year": 2016,
            "lyrics": "让我掉下眼泪的，不止昨夜的酒，让我依依不舍的，不止你的温柔",
            "genre": "民谣"
        }
    ]
    
    # 保存示例数据
    with open('data/sample_chinese_music.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info("示例数据集已创建: data/sample_chinese_music.json")
    return 'data/sample_chinese_music.json'


def demo_temporal_analysis():
    """演示时间序列分析"""
    print("🎵 中文音乐时间序列情感分析演示")
    
    # 创建分析器
    analyzer = ChineseMusicTemporalAnalyzer()
    
    # 创建示例数据集
    os.makedirs('data', exist_ok=True)
    dataset_path = create_sample_dataset()
    
    try:
        # 加载数据
        analyzer.load_music_dataset(dataset_path)
        
        # 进行时间序列分析
        results = analyzer.analyze_temporal_trends(2005, 2020)
        
        # 生成可视化
        chart_path = analyzer.create_visualization()
        print(f"📊 可视化图表已生成: {chart_path}")
        
        # 生成报告
        report_path = analyzer.generate_report()
        print(f"📋 分析报告已生成: {report_path}")
        
        # 保存结果
        results_path = analyzer.save_results()
        print(f"💾 分析结果已保存: {results_path}")
        
        print("\n✅ 时间序列分析完成！")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")


if __name__ == "__main__":
    demo_temporal_analysis() 