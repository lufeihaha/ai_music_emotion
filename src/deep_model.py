import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import librosa
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 模型参数
SAMPLE_RATE = 22050
DURATION = 30  # 秒
N_MFCC = 20
HOP_LENGTH = 512
N_SEGMENTS = 128  # 将音频分成128个片段
N_FFT = 2048

def extract_audio_features(y, sr):
    """
    提取多种音频特征
    """
    features = {}
    
    # 1. MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    features['mfcc'] = mfcc
    
    # 2. 谱质心
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features['spectral_centroids'] = spectral_centroids
    
    # 3. 谱通量
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    features['spectral_flux'] = spectral_flux
    
    # 4. 色度特征
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    features['chroma'] = chroma
    
    # 5. 零交叉率
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    features['zero_crossing_rate'] = zero_crossing_rate
    
    # 6. 谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features['spectral_bandwidth'] = spectral_bandwidth
    
    # 7. 谱衰减
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    features['spectral_rolloff'] = spectral_rolloff
    
    # 8. RMS能量
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    features['rms'] = rms
    
    return features

def apply_augmentation(y, sr):
    """
    应用数据增强
    """
    # 随机选择增强方法
    augment_type = np.random.choice(['noise', 'stretch', 'pitch', 'none'], p=[0.3, 0.3, 0.3, 0.1])
    
    if augment_type == 'noise':
        # 添加随机噪声
        noise_factor = np.random.uniform(0.001, 0.015)
        noise = np.random.normal(0, 1, len(y))
        y_aug = y + noise_factor * noise
    
    elif augment_type == 'stretch':
        # 时间拉伸/压缩
        rate = np.random.uniform(0.85, 1.15)
        y_aug = librosa.effects.time_stretch(y, rate=rate)
        
        # 确保长度一致
        if len(y_aug) > len(y):
            y_aug = y_aug[:len(y)]
        else:
            y_aug = np.pad(y_aug, (0, len(y) - len(y_aug)))
    
    elif augment_type == 'pitch':
        # 音高变换
        n_steps = np.random.randint(-4, 5)
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    else:
        # 不进行增强
        y_aug = y
    
    return y_aug

def extract_features_sequence(audio_path, augment=False):
    """
    提取特征序列
    """
    try:
        # 加载音频
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # 确保音频长度一致
        if len(y) < SAMPLE_RATE * DURATION:
            y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)))
        else:
            y = y[:SAMPLE_RATE * DURATION]
        
        # 应用数据增强
        if augment:
            y = apply_augmentation(y, sr)
        
        # 提取特征
        features = extract_audio_features(y, sr)
        
        # 将所有特征分段并组合
        feature_segments = []
        segment_size = len(features['mfcc'].T) // N_SEGMENTS
        
        for i in range(N_SEGMENTS):
            start = i * segment_size
            end = start + segment_size
            
            segment = []
            # MFCC特征
            segment.extend(np.mean(features['mfcc'][:, start:end], axis=1))
            # 谱质心
            segment.append(np.mean(features['spectral_centroids'][start:end]))
            # 谱通量
            segment.append(np.mean(features['spectral_flux'][start:end]))
            # 色度特征
            segment.extend(np.mean(features['chroma'][:, start:end], axis=1))
            # 零交叉率
            segment.append(np.mean(features['zero_crossing_rate'][start:end]))
            # 谱带宽
            segment.append(np.mean(features['spectral_bandwidth'][start:end]))
            # 谱衰减
            segment.append(np.mean(features['spectral_rolloff'][start:end]))
            # RMS能量
            segment.append(np.mean(features['rms'][start:end]))
            
            feature_segments.append(segment)
        
        return np.array(feature_segments)
    
    except Exception as e:
        logging.error(f"处理文件 {audio_path} 时出错: {str(e)}")
        return None

class TransformerBlock(layers.Layer):
    """
    Transformer编码器块
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
            kernel_regularizer=tf.keras.regularizers.l2(0.02)
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu",
                        kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            layers.Dense(embed_dim,
                        kernel_regularizer=tf.keras.regularizers.l2(0.02))
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_improved_model(input_shape, num_classes):
    """
    创建改进的混合模型，包含CNN、Transformer和双向LSTM
    """
    # 输入层
    inputs = layers.Input(shape=input_shape)
    
    # 1D卷积层用于局部特征提取
    x = layers.Conv1D(
        filters=64, kernel_size=3, padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(0.02)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # 添加残差连接的卷积块
    for _ in range(2):
        res = x
        x = layers.Conv1D(
            filters=64, kernel_size=3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Add()([x, res])
    
    # Transformer编码器
    x = TransformerBlock(64, 4, 128)(x)
    
    # 双向LSTM
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True,
                   kernel_regularizer=tf.keras.regularizers.l2(0.02),
                   recurrent_regularizer=tf.keras.regularizers.l2(0.02))
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # 全局特征池化
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # 全连接层
    x = layers.Dense(
        128, activation='gelu',
        kernel_regularizer=tf.keras.regularizers.l2(0.02)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # 输出层
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def prepare_data(augment=True):
    """
    准备训练数据
    """
    try:
        processed_dir = "data/processed"
        labels_file = os.path.join(processed_dir, "emotion_labels.csv")
        
        # 读取标签文件
        labels_df = pd.read_csv(labels_file)
        
        # 提取特征序列
        features_list = []
        valid_indices = []
        
        logging.info("开始提取音频特征...")
        for idx, row in enumerate(labels_df.iterrows()):
            audio_path = os.path.join(processed_dir, row[1]['file_path'])
            features = extract_features_sequence(audio_path, augment=False)
            
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
                
                # 对训练数据进行数据增强
                if augment:
                    # 每个样本生成2个增强版本
                    for _ in range(2):
                        aug_features = extract_features_sequence(audio_path, augment=True)
                        if aug_features is not None:
                            features_list.append(aug_features)
                            valid_indices.append(idx)
        
        # 转换为numpy数组
        X = np.array(features_list)
        y = labels_df.iloc[valid_indices]['emotion'].values
        
        # 标签编码
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # 保存标签编码器
        os.makedirs('models', exist_ok=True)
        np.save('models/label_encoder_classes.npy', label_encoder.classes_)
        
        # 特征标准化
        n_samples, n_segments, n_features = X.shape
        X_reshaped = X.reshape(n_samples * n_segments, n_features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_samples, n_segments, n_features)
        
        # 保存特征标准化器
        dump(scaler, 'models/feature_scaler.joblib')
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test, label_encoder.classes_
    
    except Exception as e:
        logging.error(f"准备数据时出错: {str(e)}")
        return None, None, None, None, None

def train_model():
    """
    训练改进的模型
    """
    try:
        # 准备数据（包含数据增强）
        X_train, X_test, y_train, y_test, class_names = prepare_data(augment=True)
        if X_train is None:
            return
        
        logging.info(f"训练集形状: {X_train.shape}")
        logging.info(f"测试集形状: {X_test.shape}")
        
        # 创建改进的模型
        input_shape = X_train.shape[1:]  # (time_steps, features)
        num_classes = len(class_names)
        model = create_improved_model(input_shape, num_classes)
        
        # 编译模型
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001,
            clipnorm=1.0  # 梯度裁剪
        )
        
        # 使用CategoricalCrossentropy和标签平滑
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)
        
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        # 创建回调函数
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            ),
            # 余弦学习率调度
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (1 + np.cos(epoch * np.pi / 100)) / 2
            )
        ]
        
        # 训练模型
        history = model.fit(
            X_train, y_train_one_hot,  # 使用one-hot编码的标签
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test_one_hot),  # 使用one-hot编码的验证集
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
        logging.info(f"\n测试集准确率: {test_acc:.4f}")
        
        # 保存模型
        model.save('models/improved_hybrid_model.keras')
        logging.info("模型已保存")
        
        # 绘制训练历史
        plot_training_history(history)
        
    except Exception as e:
        logging.error(f"训练模型时出错: {str(e)}")

def plot_training_history(history):
    """
    绘制训练历史
    """
    try:
        # 创建保存目录
        os.makedirs('models/visualizations', exist_ok=True)
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/visualizations/improved_hybrid_training_history.png')
        plt.close()
        
        logging.info("训练历史图已保存")
        
    except Exception as e:
        logging.error(f"绘制训练历史时出错: {str(e)}")

def analyze_dataset():
    """
    数据集分析和可视化
    """
    try:
        processed_dir = "data/processed"
        labels_file = os.path.join(processed_dir, "emotion_labels.csv")
        
        # 读取标签文件
        labels_df = pd.read_csv(labels_file)
        
        # 创建可视化目录
        os.makedirs('models/visualizations', exist_ok=True)
        
        # 1. 情感分布分析
        plt.figure(figsize=(10, 6))
        emotion_counts = labels_df['emotion'].value_counts()
        emotion_counts.plot(kind='bar')
        plt.title('情感类别分布')
        plt.xlabel('情感类别')
        plt.ylabel('样本数量')
        plt.tight_layout()
        plt.savefig('models/visualizations/emotion_distribution.png')
        plt.close()
        
        # 2. 特征分析
        features_list = []
        for _, row in labels_df.iterrows():
            audio_path = os.path.join(processed_dir, row['file_path'])
            features = extract_features_sequence(audio_path, augment=False)
            if features is not None:
                features_list.append(features.mean(axis=0))  # 对时间维度取平均
                
        features_df = pd.DataFrame(features_list)
        
        # 特征相关性热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(features_df.corr(), annot=False, cmap='coolwarm')
        plt.title('特征相关性分析')
        plt.tight_layout()
        plt.savefig('models/visualizations/feature_correlation.png')
        plt.close()
        
        # 3. 特征重要性分析（使用随机森林）
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features_df, labels_df['emotion'])
        
        # 绘制特征重要性
        plt.figure(figsize=(12, 6))
        feature_importance = pd.Series(rf.feature_importances_)
        feature_importance.sort_values(ascending=False).head(10).plot(kind='bar')
        plt.title('Top 10 重要特征')
        plt.xlabel('特征索引')
        plt.ylabel('重要性得分')
        plt.tight_layout()
        plt.savefig('models/visualizations/feature_importance.png')
        plt.close()
        
        logging.info("数据分析和可视化完成")
        
    except Exception as e:
        logging.error(f"数据分析时出错: {str(e)}")

def evaluate_model(model, X_test, y_test, class_names):
    """
    详细的模型评估
    """
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        # 预测测试集
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # 1. 混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.tight_layout()
        plt.savefig('models/visualizations/confusion_matrix.png')
        plt.close()
        
        # 2. 分类报告
        report = classification_report(y_test_classes, y_pred_classes,
                                    target_names=class_names,
                                    output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('models/classification_report.csv')
        
        # 3. ROC曲线（多分类）
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
        
        for i, color in zip(range(len(class_names)), colors):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('多分类ROC曲线')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('models/visualizations/roc_curves.png')
        plt.close()
        
        logging.info("模型评估完成，结果已保存")
        
    except Exception as e:
        logging.error(f"模型评估时出错: {str(e)}")

if __name__ == "__main__":
    train_model()
    analyze_dataset()