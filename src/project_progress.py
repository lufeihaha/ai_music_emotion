"""
项目阶段进度追踪与自动评估
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import load

# 阶段1：数据准备与探索

def data_exploration():
    processed_dir = "data/processed"
    labels_file = os.path.join(processed_dir, "emotion_labels.csv")
    labels_df = pd.read_csv(labels_file)
    os.makedirs('models/visualizations', exist_ok=True)
    # 1. 情感类别分布
    plt.figure(figsize=(10, 6))
    labels_df['emotion'].value_counts().plot(kind='bar')
    plt.title('情感类别分布')
    plt.xlabel('情感类别')
    plt.ylabel('样本数量')
    plt.tight_layout()
    plt.savefig('models/visualizations/emotion_distribution.png')
    plt.close()
    # 2. 音频时长分布（如有duration列）
    if 'duration' in labels_df.columns:
        plt.figure(figsize=(8, 5))
        labels_df['duration'].hist(bins=30)
        plt.title('音频时长分布')
        plt.xlabel('时长（秒）')
        plt.ylabel('数量')
        plt.tight_layout()
        plt.savefig('models/visualizations/audio_duration.png')
        plt.close()
    # 3. 不同情感类别的特征分布（如MFCC均值箱线图）
    features_path = 'data/features.npy'
    if os.path.exists(features_path):
        import numpy as np
        X = np.load(features_path)
        mfcc_mean = X.mean(axis=(1,2)) if X.ndim == 3 else X.mean(axis=1)
        labels_df['mfcc_mean'] = mfcc_mean
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='emotion', y='mfcc_mean', data=labels_df)
        plt.title('不同情感类别的MFCC均值分布')
        plt.tight_layout()
        plt.savefig('models/visualizations/mfcc_mean_by_emotion.png')
        plt.close()
    print("[阶段1] 数据探索与可视化完成")

# 阶段2：特征工程

def feature_engineering():
    features_path = 'data/features.npy'
    labels_file = 'data/processed/emotion_labels.csv'
    if not os.path.exists(features_path):
        print("未找到特征文件，跳过特征工程可视化")
        return
    import numpy as np
    X = np.load(features_path)
    labels_df = pd.read_csv(labels_file)
    y = labels_df['emotion'].values
    # 降维可视化
    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)
    plt.figure(figsize=(8, 6))
    for emo in set(y):
        idx = y == emo
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=emo, alpha=0.6)
    plt.legend()
    plt.title('PCA降维后的特征分布')
    plt.tight_layout()
    plt.savefig('models/visualizations/feature_pca.png')
    plt.close()
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_flat)
    plt.figure(figsize=(8, 6))
    for emo in set(y):
        idx = y == emo
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=emo, alpha=0.6)
    plt.legend()
    plt.title('t-SNE降维后的特征分布')
    plt.tight_layout()
    plt.savefig('models/visualizations/feature_tsne.png')
    plt.close()
    # 特征相关性热力图
    if X_flat.shape[1] < 100:  # 特征数太多会很慢
        features_df = pd.DataFrame(X_flat)
        plt.figure(figsize=(12, 8))
        sns.heatmap(features_df.corr(), annot=False, cmap='coolwarm')
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('models/visualizations/feature_correlation.png')
        plt.close()
    print("[阶段2] 特征工程可视化完成")

# 阶段3：模型开发

def model_development_report():
    model_dir = 'models'
    files = os.listdir(model_dir)
    print("[阶段3] 已保存模型：")
    for f in files:
        if f.endswith('.pkl') or f.endswith('.keras'):
            print("  -", f)
    print("如需详细模型结构，请查看 src/deep_model.py 或 src/emotion_classifier.py")

# 阶段4：训练与优化

def model_evaluation_report():
    report_path = 'models/classification_report.csv'
    cm_path = 'models/visualizations/confusion_matrix.png'
    if os.path.exists(report_path):
        df = pd.read_csv(report_path, index_col=0)
        print("[阶段4] 分类报告：")
        print(df[['precision', 'recall', 'f1-score', 'support']])
    else:
        print("未找到分类报告，请先训练模型并评估")
    if os.path.exists(cm_path):
        print("混淆矩阵图片已生成：models/visualizations/confusion_matrix.png")
    else:
        print("未找到混淆矩阵图片")

# 阶段5：系统集成与测试

def system_integration_check():
    app_path = 'src/app.py'
    if os.path.exists(app_path):
        print("[阶段5] 检测到Web系统入口 src/app.py，可进一步开发UI和API接口")
    else:
        print("未检测到Web系统入口文件，请补充")

if __name__ == "__main__":
    print("==== 项目阶段进度追踪 ====")
    data_exploration()
    feature_engineering()
    model_development_report()
    model_evaluation_report()
    system_integration_check()
    print("==== 阶段追踪完毕 ====")
