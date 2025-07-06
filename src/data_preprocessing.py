import os
import shutil
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def create_emotion_mapping():
    """创建音乐类型到情感的映射"""
    return {
        'classical': 'calm',
        'jazz': 'calm',
        'blues': 'melancholic',
        'metal': 'energetic',
        'rock': 'energetic',
        'disco': 'happy',
        'pop': 'happy',
        'hiphop': 'energetic',
        'reggae': 'calm',
        'country': 'melancholic'
    }

def process_dataset(raw_dir: str, processed_dir: str):
    """处理数据集，重组织文件结构"""
    # 创建情感标签目录
    emotion_dirs = ['calm', 'energetic', 'happy', 'melancholic']
    for emotion in emotion_dirs:
        os.makedirs(os.path.join(processed_dir, emotion), exist_ok=True)
    
    # 获取音乐类型到情感的映射
    emotion_mapping = create_emotion_mapping()
    
    # 处理每个音乐类型目录
    for genre in emotion_mapping.keys():
        genre_dir = os.path.join(raw_dir, genre)
        if not os.path.exists(genre_dir):
            logging.warning(f"找不到目录: {genre_dir}")
            continue
            
        emotion = emotion_mapping[genre]
        target_dir = os.path.join(processed_dir, emotion)
        
        # 复制音频文件
        for file_name in os.listdir(genre_dir):
            if not file_name.endswith('.wav'):
                continue
                
            source_path = os.path.join(genre_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            
            try:
                shutil.copy2(source_path, target_path)
                logging.info(f"复制文件: {source_path} -> {target_path}")
            except Exception as e:
                logging.error(f"复制文件失败: {str(e)}")

def create_emotion_labels(processed_dir: str):
    """创建情感标签CSV文件"""
    data = []
    
    for emotion in ['calm', 'energetic', 'happy', 'melancholic']:
        emotion_dir = os.path.join(processed_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        for file_name in os.listdir(emotion_dir):
            if file_name.endswith('.wav'):
                data.append({
                    'file_path': os.path.join(emotion, file_name),
                    'emotion': emotion
                })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    csv_path = os.path.join(processed_dir, 'emotion_labels.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"创建标签文件: {csv_path}")

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    raw_dir = os.path.join(project_dir, 'data', 'raw')
    processed_dir = os.path.join(project_dir, 'data', 'processed')
    
    # 创建处理后的数据目录
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # 处理数据集
        process_dataset(raw_dir, processed_dir)
        
        # 创建标签文件
        create_emotion_labels(processed_dir)
        
        logging.info("数据预处理完成")
        
    except Exception as e:
        logging.error(f"数据预处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 