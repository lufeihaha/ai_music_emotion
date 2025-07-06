import joblib
import numpy as np

def main():
    # 加载模型
    print("加载随机森林模型...")
    model = joblib.load('models/random_forest.pkl')
    
    print("\n模型信息:")
    print(f"模型类型: {type(model).__name__}")
    print(f"树的数量: {model.n_estimators}")
    print(f"特征数量: {model.n_features_in_}")
    print(f"类别数量: {len(model.classes_)}")
    print(f"类别标签: {model.classes_}")
    
    # 获取特征重要性
    importances = model.feature_importances_
    top_n = 10
    indices = np.argsort(importances)[::-1][:top_n]
    
    print(f"\n前{top_n}个最重要的特征:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. 特征 {idx}: {importances[idx]:.4f}")
    
    # 加载标准化器
    print("\n加载标准化器...")
    scaler = joblib.load('models/scaler.pkl')
    
    print("\n标准化器信息:")
    print(f"标准化器类型: {type(scaler).__name__}")
    print(f"特征数量: {scaler.n_features_in_}")
    print(f"特征均值范围: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"特征标准差范围: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

if __name__ == "__main__":
    main() 