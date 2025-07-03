import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import logging
from audio_processor import AudioProcessor
from data_augmentation import DataAugmentor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model(data_dir: str, model_save_path: str):
    """
    训练音乐情感分类模型
    
    Args:
        data_dir: 数据集目录
        model_save_path: 模型保存路径
    """
    # 读取情感标签
    labels_path = os.path.join(os.path.dirname(data_dir), "processed", "emotion_labels.csv")
    if not os.path.exists(labels_path):
        raise ValueError(f"找不到标签文件: {labels_path}")

    labels_df = pd.read_csv(labels_path)

    # 初始化音频处理器
    processor = AudioProcessor(sr=22050, duration=30)

    print("开始提取特征和标签...")
    features = []
    emotions = []
    print(f"处理 {len(labels_df)} 个音频文件...")

    for idx, row in labels_df.iterrows():
        if idx > 0 and idx % 100 == 0:
            print(f"已处理 {idx} 个文件...")

        mp3_path = row['mp3_path']
        emotion = row['emotion']

        # 跳过中性样本，只使用有明确情感标签的样本
        if emotion == 'neutral':
            continue

        # 构建完整的文件路径
        file_path = os.path.join(data_dir, mp3_path)
        if not os.path.exists(file_path):
            print(f"找不到文件: {file_path}")
            continue

        try:
            feature_vector = processor.process_audio_file(file_path)
            features.append(feature_vector)
            emotions.append(emotion)
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {str(e)}")
            continue

    X = np.array(features)
    y = np.array(emotions)
    
    # 数据增强
    print("开始数据增强...")
    augmentor = DataAugmentor(sr=22050)
    
    # 计算每个类别的样本数量
    unique, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    
    # 确定每个类别需要增强的数量
    augmentations_per_class = {
        label: max_count - count 
        for label, count in zip(unique, counts)
    }
    
    # 进行数据增强
    X_augmented, y_augmented = augmentor.augment_dataset(X, y, augmentations_per_class)
    
    # 数据标准化
    print("数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_augmented)
    
    # 保存数据标准化器
    joblib.dump(scaler, os.path.join(os.path.dirname(model_save_path), 'scaler.pkl'))
    
    # 分割训练集和测试集
    print("分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_augmented, test_size=0.2, random_state=42
    )
    
    # 训练模型
    print("开始训练模型...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 评估模型
    print("评估模型...")
    y_pred = model.predict(X_test)
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=genres))
    
    # 保存模型
    print(f"保存模型到 {model_save_path}")
    joblib.dump(model, model_save_path)
    
    return model, scaler

if __name__ == '__main__':
    try:
        # 设置路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(project_dir, "data", "raw")
        model_dir = os.path.join(project_dir, "models")
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 训练模型
        model_path = os.path.join(model_dir, "music_emotion_classifier.pkl")
        model, scaler = train_model(data_dir, model_path)
        print("模型训练完成！")
        
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}")
        raise