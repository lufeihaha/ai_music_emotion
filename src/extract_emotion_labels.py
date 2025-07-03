import pandas as pd
import numpy as np
import os

def map_to_emotion(row):
    """
    将特征映射到情感标签
    
    Args:
        row: DataFrame的一行
        
    Returns:
        情感标签
    """
    # 定义情感特征组合
    emotion_features = {
        'happy': ['happy', 'upbeat', 'funky', 'fun'],
        'sad': ['sad', 'slow', 'dark', 'melancholy'],
        'energetic': ['fast', 'loud', 'heavy', 'hard', 'quick', 'beats'],
        'calm': ['soft', 'quiet', 'calm', 'peaceful', 'light', 'mellow'],
        'aggressive': ['metal', 'hard rock', 'heavy metal', 'punk', 'aggressive']
    }
    
    # 计算每个情感类别的分数
    scores = {emotion: 0 for emotion in emotion_features}
    for emotion, features in emotion_features.items():
        for feature in features:
            if feature in row and row[feature] == 1:
                scores[emotion] += 1
    
    # 找出得分最高的情感
    if max(scores.values()) == 0:
        return 'neutral'
    return max(scores.items(), key=lambda x: x[1])[0]

def extract_emotion_labels(csv_path: str) -> pd.DataFrame:
    """
    从CSV文件中提取音乐情感标签
    
    Args:
        csv_path: 标注CSV文件路径
        
    Returns:
        包含音乐片段ID和情感标签的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path, sep='\t')
    
    # 为每个片段分配情感标签
    emotions = []
    for _, row in df.iterrows():
        emotion = map_to_emotion(row)
        emotions.append(emotion)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'clip_id': df['clip_id'],
        'mp3_path': df['mp3_path'],
        'emotion': emotions
    })
    
    return result_df

def save_emotion_labels(csv_path: str, output_path: str):
    """
    保存处理后的情感标签
    
    Args:
        csv_path: 输入CSV文件路径
        output_path: 输出CSV文件路径
    """
    df = extract_emotion_labels(csv_path)
    df.to_csv(output_path, index=False)
    print(f"标签已保存到: {output_path}")
    
    # 打印每个情感类别的数量
    emotion_counts = df['emotion'].value_counts()
    print("\n情感分布:")
    print(emotion_counts)

if __name__ == "__main__":
    # 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    input_csv = os.path.join(project_dir, "data", "raw", "annotations_final.csv")
    output_csv = os.path.join(project_dir, "data", "processed", "emotion_labels.csv")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 处理并保存标签
    save_emotion_labels(input_csv, output_csv)
