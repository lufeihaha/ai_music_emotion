#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文音乐情感数据集下载和集成工具
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from urllib.parse import urljoin
import zipfile
import tarfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseMusicDatasetDownloader:
    """中文音乐数据集下载器"""
    
    def __init__(self, base_dir: str = "data/chinese_datasets"):
        """
        初始化下载器
        
        Args:
            base_dir: 数据集保存的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集配置
        self.datasets_config = {
            "PMEmo": {
                "url": "https://github.com/HuiZhangDB/PMEmo",
                "description": "PMEmo数据集 - 794首流行音乐，情感和生理信号标注",
                "emotions": ["arousal", "valence", "static_emotion", "dynamic_emotion"],
                "format": "csv",
                "size": "794 songs"
            },
            "PSIC3839": {
                "url": "https://github.com/xl2218066/PSIC3839", 
                "description": "PSIC3839数据集 - 3839首中国流行歌曲",
                "emotions": ["arousal", "valence", "depth"],
                "format": "xlsx",
                "size": "3839 songs"
            },
            "CCMusic": {
                "url": "https://huggingface.co/datasets/ccmusic",
                "description": "CCMusic数据集 - 中文音乐信息检索综合数据集",
                "emotions": ["multiple_categories"],
                "format": "multiple",
                "size": "multiple datasets"
            }
        }
    
    def download_pmemo_dataset(self) -> bool:
        """
        下载PMEmo数据集
        
        Returns:
            bool: 下载是否成功
        """
        try:
            logger.info("开始下载PMEmo数据集...")
            
            pmemo_dir = self.base_dir / "PMEmo"
            pmemo_dir.mkdir(exist_ok=True)
            
            # PMEmo数据集的直接下载链接（需要根据实际情况调整）
            dataset_urls = {
                "annotations.csv": "https://raw.githubusercontent.com/HuiZhangDB/PMEmo/master/dataset/annotations.csv",
                "static_annotations.csv": "https://raw.githubusercontent.com/HuiZhangDB/PMEmo/master/dataset/static_annotations.csv",
                "dynamic_annotations.csv": "https://raw.githubusercontent.com/HuiZhangDB/PMEmo/master/dataset/dynamic_annotations.csv"
            }
            
            for filename, url in dataset_urls.items():
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(pmemo_dir / filename, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"成功下载: {filename}")
                    else:
                        logger.warning(f"无法下载 {filename}: HTTP {response.status_code}")
                except Exception as e:
                    logger.error(f"下载 {filename} 失败: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"下载PMEmo数据集失败: {str(e)}")
            return False
    
    def download_psic3839_dataset(self) -> bool:
        """
        下载PSIC3839数据集
        
        Returns:
            bool: 下载是否成功
        """
        try:
            logger.info("开始下载PSIC3839数据集...")
            
            psic_dir = self.base_dir / "PSIC3839"
            psic_dir.mkdir(exist_ok=True)
            
            # PSIC3839数据集的下载链接
            dataset_urls = {
                "general_data.xlsx": "https://raw.githubusercontent.com/xl2218066/PSIC3839/master/final%20upload/general_data.xlsx",
                "sub01.xlsx": "https://raw.githubusercontent.com/xl2218066/PSIC3839/master/final%20upload/sub01.xlsx",
                "sub02.xlsx": "https://raw.githubusercontent.com/xl2218066/PSIC3839/master/final%20upload/sub02.xlsx",
                "sub03.xlsx": "https://raw.githubusercontent.com/xl2218066/PSIC3839/master/final%20upload/sub03.xlsx",
                "sub04.xlsx": "https://raw.githubusercontent.com/xl2218066/PSIC3839/master/final%20upload/sub04.xlsx",
                "sub05.xlsx": "https://raw.githubusercontent.com/xl2218066/PSIC3839/master/final%20upload/sub05.xlsx"
            }
            
            for filename, url in dataset_urls.items():
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(psic_dir / filename, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"成功下载: {filename}")
                    else:
                        logger.warning(f"无法下载 {filename}: HTTP {response.status_code}")
                except Exception as e:
                    logger.error(f"下载 {filename} 失败: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"下载PSIC3839数据集失败: {str(e)}")
            return False
    
    def create_chinese_emotion_mapping(self) -> Dict[str, str]:
        """
        创建中文情感映射表
        
        Returns:
            Dict[str, str]: 中文情感到英文情感的映射
        """
        emotion_mapping = {
            # 基础情感
            "快乐": "happy",
            "高兴": "happy", 
            "愉快": "happy",
            "欢乐": "happy",
            "喜悦": "happy",
            
            "悲伤": "melancholic",
            "忧伤": "melancholic",
            "难过": "melancholic",
            "哀伤": "melancholic",
            "沮丧": "melancholic",
            
            "平静": "calm",
            "安静": "calm",
            "宁静": "calm",
            "祥和": "calm",
            "放松": "calm",
            
            "激动": "energetic",
            "兴奋": "energetic",
            "活跃": "energetic",
            "充满活力": "energetic",
            "热情": "energetic",
            
            # 扩展情感
            "怀念": "nostalgic",
            "思念": "nostalgic",
            "回忆": "nostalgic",
            "眷恋": "nostalgic",
            
            "浪漫": "romantic",
            "温柔": "romantic",
            "甜蜜": "romantic",
            "爱情": "romantic",
            
            "戏剧性": "dramatic",
            "紧张": "dramatic",
            "激烈": "dramatic",
            
            "神秘": "mysterious",
            "迷幻": "mysterious",
            "深沉": "mysterious"
        }
        
        return emotion_mapping
    
    def process_chinese_annotations(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        处理中文数据集的标注
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            Optional[pd.DataFrame]: 处理后的标注数据
        """
        try:
            if dataset_name == "PMEmo":
                return self._process_pmemo_annotations()
            elif dataset_name == "PSIC3839":
                return self._process_psic3839_annotations()
            else:
                logger.warning(f"不支持的数据集: {dataset_name}")
                return None
                
        except Exception as e:
            logger.error(f"处理 {dataset_name} 标注失败: {str(e)}")
            return None
    
    def _process_pmemo_annotations(self) -> Optional[pd.DataFrame]:
        """处理PMEmo数据集标注"""
        try:
            pmemo_dir = self.base_dir / "PMEmo"
            
            # 检查文件是否存在
            static_file = pmemo_dir / "static_annotations.csv"
            if not static_file.exists():
                logger.warning("PMEmo静态标注文件不存在")
                return None
            
            # 读取静态标注
            df = pd.read_csv(static_file)
            
            # 标准化列名
            df_processed = pd.DataFrame({
                'song_id': df.get('song_id', range(len(df))),
                'arousal': df.get('arousal_mean', df.get('arousal', 0)),
                'valence': df.get('valence_mean', df.get('valence', 0)),
                'emotion': df.get('emotion', 'unknown'),
                'dataset': 'PMEmo'
            })
            
            # 根据arousal和valence映射到我们的4类情感
            df_processed['mapped_emotion'] = df_processed.apply(
                lambda row: self._map_av_to_emotion(row['arousal'], row['valence']), 
                axis=1
            )
            
            return df_processed
            
        except Exception as e:
            logger.error(f"处理PMEmo标注失败: {str(e)}")
            return None
    
    def _process_psic3839_annotations(self) -> Optional[pd.DataFrame]:
        """处理PSIC3839数据集标注"""
        try:
            psic_dir = self.base_dir / "PSIC3839"
            
            # 检查文件是否存在
            general_file = psic_dir / "general_data.xlsx"
            if not general_file.exists():
                logger.warning("PSIC3839综合数据文件不存在")
                return None
            
            # 读取综合数据
            df = pd.read_excel(general_file)
            
            # 标准化列名
            df_processed = pd.DataFrame({
                'song_id': df.get('id', range(len(df))),
                'arousal': df.get('arousal_mean', 0),
                'valence': df.get('valence_mean', 0),
                'depth': df.get('depth_mean', 0),
                'dataset': 'PSIC3839'
            })
            
            # 根据arousal和valence映射到我们的4类情感
            df_processed['mapped_emotion'] = df_processed.apply(
                lambda row: self._map_av_to_emotion(row['arousal'], row['valence']), 
                axis=1
            )
            
            return df_processed
            
        except Exception as e:
            logger.error(f"处理PSIC3839标注失败: {str(e)}")
            return None
    
    def _map_av_to_emotion(self, arousal: float, valence: float) -> str:
        """
        根据arousal和valence映射到4类情感
        
        Args:
            arousal: 唤醒度
            valence: 效价
            
        Returns:
            str: 映射的情感类别
        """
        # 标准化到0-1范围（假设输入是1-9范围）
        if arousal > 1:
            arousal = (arousal - 1) / 8
        if valence > 1:
            valence = (valence - 1) / 8
            
        # 四象限映射
        if arousal > 0.5 and valence > 0.5:
            return "happy"  # 高唤醒高效价 -> 快乐
        elif arousal > 0.5 and valence <= 0.5:
            return "energetic"  # 高唤醒低效价 -> 充满活力
        elif arousal <= 0.5 and valence > 0.5:
            return "calm"  # 低唤醒高效价 -> 平静
        else:
            return "melancholic"  # 低唤醒低效价 -> 忧郁
    
    def integrate_chinese_datasets(self) -> bool:
        """
        整合所有中文数据集
        
        Returns:
            bool: 整合是否成功
        """
        try:
            logger.info("开始整合中文数据集...")
            
            all_datasets = []
            
            # 处理PMEmo数据集
            pmemo_data = self.process_chinese_annotations("PMEmo")
            if pmemo_data is not None:
                all_datasets.append(pmemo_data)
                logger.info(f"PMEmo数据集: {len(pmemo_data)} 条记录")
            
            # 处理PSIC3839数据集
            psic_data = self.process_chinese_annotations("PSIC3839")
            if psic_data is not None:
                all_datasets.append(psic_data)
                logger.info(f"PSIC3839数据集: {len(psic_data)} 条记录")
            
            if not all_datasets:
                logger.warning("没有可用的中文数据集")
                return False
            
            # 合并所有数据集
            combined_df = pd.concat(all_datasets, ignore_index=True)
            
            # 保存整合后的数据
            output_file = self.base_dir / "chinese_music_emotions.csv"
            combined_df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"中文数据集整合完成: {len(combined_df)} 条记录")
            logger.info(f"保存至: {output_file}")
            
            # 打印统计信息
            self._print_dataset_stats(combined_df)
            
            return True
            
        except Exception as e:
            logger.error(f"整合中文数据集失败: {str(e)}")
            return False
    
    def _print_dataset_stats(self, df: pd.DataFrame):
        """打印数据集统计信息"""
        logger.info("=== 中文数据集统计信息 ===")
        logger.info(f"总记录数: {len(df)}")
        
        if 'dataset' in df.columns:
            logger.info("各数据集分布:")
            for dataset, count in df['dataset'].value_counts().items():
                logger.info(f"  {dataset}: {count} 条")
        
        if 'mapped_emotion' in df.columns:
            logger.info("情感类别分布:")
            for emotion, count in df['mapped_emotion'].value_counts().items():
                logger.info(f"  {emotion}: {count} 条")
    
    def download_all_datasets(self) -> bool:
        """
        下载所有可用的中文数据集
        
        Returns:
            bool: 下载是否成功
        """
        success_count = 0
        total_count = 0
        
        logger.info("开始下载所有中文音乐数据集...")
        
        # 下载PMEmo
        total_count += 1
        if self.download_pmemo_dataset():
            success_count += 1
        
        # 下载PSIC3839
        total_count += 1
        if self.download_psic3839_dataset():
            success_count += 1
        
        logger.info(f"数据集下载完成: {success_count}/{total_count} 成功")
        
        return success_count > 0

def main():
    """主函数"""
    print("🎵 中文音乐情感数据集下载器")
    print("=" * 50)
    
    downloader = ChineseMusicDatasetDownloader()
    
    print("\n📋 可用的中文音乐数据集:")
    for name, config in downloader.datasets_config.items():
        print(f"  {name}: {config['description']}")
        print(f"    规模: {config['size']}")
        print(f"    情感维度: {config['emotions']}")
        print()
    
    choice = input("是否开始下载数据集? (y/n): ").lower()
    if choice == 'y':
        # 下载数据集
        if downloader.download_all_datasets():
            print("✅ 数据集下载完成!")
            
            # 整合数据集
            if downloader.integrate_chinese_datasets():
                print("✅ 数据集整合完成!")
                print(f"📁 数据保存在: {downloader.base_dir}")
            else:
                print("❌ 数据集整合失败")
        else:
            print("❌ 数据集下载失败")
    else:
        print("操作已取消")

if __name__ == "__main__":
    main() 