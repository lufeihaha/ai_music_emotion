import os
import pandas as pd

def create_labels():
    # 设置路径
    processed_dir = "data/processed"
    labels_file = os.path.join(processed_dir, "emotion_labels.csv")
    
    # 初始化列表存储文件路径和标签
    file_paths = []
    emotions = []
    
    # 遍历情感目录
    for emotion in ['calm', 'energetic', 'happy', 'melancholic']:
        emotion_dir = os.path.join(processed_dir, emotion)
        if os.path.exists(emotion_dir):
            # 遍历该情感目录下的所有WAV文件
            for file_name in os.listdir(emotion_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(emotion, file_name)
                    file_paths.append(file_path)
                    emotions.append(emotion)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'file_path': file_paths,
        'emotion': emotions
    })
    
    # 保存到CSV文件
    df.to_csv(labels_file, index=False)
    print(f"已创建标签文件，共 {len(df)} 个样本")
    print("\n前5个样本:")
    print(df.head())

if __name__ == "__main__":
    create_labels() 