"""
特征选择与降维工具
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def feature_importance_selection():
    features_path = 'data/features.npy'
    labels_file = 'data/processed/emotion_labels.csv'
    if not (os.path.exists(features_path) and os.path.exists(labels_file)):
        print("缺少特征或标签文件")
        return
    X = np.load(features_path)
    labels_df = pd.read_csv(labels_file)
    y = labels_df['emotion'].values
    X_flat = X.reshape(X.shape[0], -1)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_flat, y)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10, 5))
    plt.bar(range(20), importances[idx])
    plt.title('Top 20 特征重要性')
    plt.xlabel('特征索引')
    plt.ylabel('重要性')
    plt.tight_layout()
    plt.savefig('models/visualizations/top20_feature_importance.png')
    plt.close()
    print("特征重要性图已保存")

def pca_dim_reduction(n_components=20):
    features_path = 'data/features.npy'
    if not os.path.exists(features_path):
        print("缺少特征文件")
        return
    X = np.load(features_path)
    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_flat)
    np.save('data/features_pca.npy', X_pca)
    print(f"PCA降维后特征 shape: {X_pca.shape}, 已保存 data/features_pca.npy")

if __name__ == "__main__":
    feature_importance_selection()
    pca_dim_reduction()
