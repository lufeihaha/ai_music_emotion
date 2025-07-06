#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class DataEnhancer:
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 30
        
    def augment_audio(self, audio_path, output_dir, augment_count=3):
        """
        对单个音频文件进行数据增强
        """
        try:
            # 加载原始音频
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            augmented_files = []
            
            # 1. 调整音调 (Pitch Shifting)
            for i, pitch_shift in enumerate([-2, -1, 1, 2]):
                y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
                output_path = os.path.join(output_dir, f"{base_name}_pitch_{pitch_shift:+d}.wav")
                sf.write(output_path, y_pitched, sr)
                augmented_files.append(output_path)
            
            # 2. 调整速度 (Time Stretching)
            for i, stretch_factor in enumerate([0.8, 0.9, 1.1, 1.2]):
                y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
                # 确保长度一致
                if len(y_stretched) > len(y):
                    y_stretched = y_stretched[:len(y)]
                else:
                    y_stretched = np.pad(y_stretched, (0, len(y) - len(y_stretched)), 'constant')
                
                output_path = os.path.join(output_dir, f"{base_name}_stretch_{stretch_factor:.1f}.wav")
                sf.write(output_path, y_stretched, sr)
                augmented_files.append(output_path)
            
            # 3. 添加噪声
            for i, noise_factor in enumerate([0.005, 0.01]):
                noise = np.random.randn(len(y))
                y_noisy = y + noise_factor * noise
                output_path = os.path.join(output_dir, f"{base_name}_noise_{noise_factor:.3f}.wav")
                sf.write(output_path, y_noisy, sr)
                augmented_files.append(output_path)
            
            return augmented_files
            
        except Exception as e:
            logging.error(f"增强音频 {audio_path} 失败: {str(e)}")
            return []
    
    def enhance_dataset(self, data_dir="data/processed"):
        """
        增强整个数据集
        """
        enhanced_data = []
        
        for emotion_dir in os.listdir(data_dir):
            emotion_path = os.path.join(data_dir, emotion_dir)
            if not os.path.isdir(emotion_path):
                continue
                
            logging.info(f"处理情感类别: {emotion_dir}")
            
            # 创建增强数据输出目录
            enhanced_dir = os.path.join(data_dir, f"{emotion_dir}_enhanced")
            os.makedirs(enhanced_dir, exist_ok=True)
            
            # 获取原始文件
            original_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            
            # 对每个文件进行增强
            for file_name in tqdm(original_files[:50], desc=f"增强 {emotion_dir}"):  # 限制数量避免过多
                file_path = os.path.join(emotion_path, file_name)
                augmented_files = self.augment_audio(file_path, enhanced_dir)
                
                # 记录增强后的文件
                for aug_file in augmented_files:
                    enhanced_data.append({
                        'file_path': aug_file,
                        'emotion': emotion_dir,
                        'augmented': True,
                        'original_file': file_path
                    })
        
        # 保存增强数据清单
        df = pd.DataFrame(enhanced_data)
        df.to_csv(os.path.join(data_dir, 'enhanced_data_list.csv'), index=False)
        
        logging.info(f"数据增强完成，共生成 {len(enhanced_data)} 个增强样本")
        return df

    def balance_dataset(self, data_dir="data/processed"):
        """
        平衡数据集中各情感类别的样本数量
        """
        emotion_counts = {}
        
        # 统计各类别样本数量
        for emotion_dir in os.listdir(data_dir):
            emotion_path = os.path.join(data_dir, emotion_dir)
            if os.path.isdir(emotion_path) and not emotion_dir.endswith('_enhanced'):
                files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
                emotion_counts[emotion_dir] = len(files)
        
        logging.info("当前数据分布:")
        for emotion, count in emotion_counts.items():
            logging.info(f"  {emotion}: {count} 个样本")
        
        # 找到最多的类别作为目标数量
        target_count = max(emotion_counts.values())
        logging.info(f"目标样本数量: {target_count}")
        
        # 为样本不足的类别生成更多增强数据
        for emotion, current_count in emotion_counts.items():
            if current_count < target_count:
                needed = target_count - current_count
                logging.info(f"{emotion} 需要增加 {needed} 个样本")
                
                emotion_path = os.path.join(data_dir, emotion)
                enhanced_dir = os.path.join(data_dir, f"{emotion}_balanced")
                os.makedirs(enhanced_dir, exist_ok=True)
                
                original_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
                
                # 重复使用原始文件进行增强直到达到目标数量
                generated = 0
                while generated < needed:
                    for file_name in original_files:
                        if generated >= needed:
                            break
                        file_path = os.path.join(emotion_path, file_name)
                        self.augment_audio(file_path, enhanced_dir, augment_count=1)
                        generated += 1

if __name__ == "__main__":
    enhancer = DataEnhancer()
    
    print("🔄 开始数据增强...")
    enhancer.enhance_dataset()
    
    print("⚖️ 开始数据平衡...")
    enhancer.balance_dataset()
    
    print("✅ 数据优化完成!") 