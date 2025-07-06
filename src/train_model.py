import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 设置中文字体
def set_chinese_font():
    try:
        # 尝试设置微软雅黑字体
        font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
        plt.rcParams['font.family'] = ['Microsoft YaHei']
        return font
    except:
        try:
            # 尝试设置宋体
            font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
            plt.rcParams['font.family'] = ['SimSun']
            return font
        except:
            logging.warning("无法加载中文字体，将使用默认字体")
            return None

def load_data():
    """
    加载特征和标签数据
    """
    try:
        # 加载特征
        features = np.load('data/features.npy')
        logging.info(f"加载特征成功，形状: {features.shape}")
        
        # 加载标签
        labels_df = pd.read_csv('data/processed/emotion_labels.csv')
        labels = labels_df['emotion'].values
        logging.info(f"加载标签成功，形状: {labels.shape}")
        
        return features, labels
    except Exception as e:
        logging.error(f"加载数据失败: {str(e)}")
        return None, None

def preprocess_data(features, labels):
    """
    数据预处理：标准化特征并划分训练集和测试集
    """
    try:
        # 如果特征是3维的，需要重塑为2维
        if len(features.shape) == 3:
            # 将 (n_samples, n_segments, n_features) 重塑为 (n_samples, n_segments * n_features)
            n_samples, n_segments, n_features = features.shape
            features = features.reshape(n_samples, n_segments * n_features)
            logging.info(f"特征从3D {(n_samples, n_segments, n_features)} 重塑为2D {features.shape}")
        
        # 标签编码
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        logging.info(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("特征标准化完成")
        
        # 保存标准化器和标签编码器
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(label_encoder, 'models/label_encoder.pkl')
        logging.info("标准化器和标签编码器已保存")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder
    except Exception as e:
        logging.error(f"数据预处理失败: {str(e)}")
        return None, None, None, None, None

def create_models():
    """
    创建多个模型
    """
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
    }
    return models

def plot_confusion_matrix(y_true, y_pred, label_encoder, model_name):
    """
    绘制并保存混淆矩阵
    """
    try:
        # 设置中文字体
        font = set_chinese_font()
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 获取标签
        labels = ['平静', '充满活力', '快乐', '忧郁']
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels)
        
        # 设置标题和标签
        if font:
            plt.title(f'{model_name} 混淆矩阵', fontproperties=font, fontsize=14)
            plt.xlabel('预测标签', fontproperties=font, fontsize=12)
            plt.ylabel('真实标签', fontproperties=font, fontsize=12)
        else:
            plt.title(f'{model_name} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        os.makedirs('models/visualizations', exist_ok=True)
        plt.savefig(f'models/visualizations/{model_name}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"{model_name} 混淆矩阵已保存")
    except Exception as e:
        logging.error(f"绘制混淆矩阵失败: {str(e)}")

def plot_feature_importance(model, feature_importance, top_n=20):
    """
    绘制特征重要性
    """
    try:
        # 获取前N个最重要的特征
        indices = np.argsort(feature_importance)[::-1][:top_n]
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        plt.title('特征重要性 Top 20')
        plt.bar(range(top_n), feature_importance[indices])
        plt.xticks(range(top_n), indices, rotation=45)
        plt.xlabel('特征索引')
        plt.ylabel('重要性')
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('models/visualizations/feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("特征重要性图已保存")
    except Exception as e:
        logging.error(f"绘制特征重要性失败: {str(e)}")

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    """
    训练和评估多个模型
    """
    try:
        # 创建模型
        models = create_models()
        best_model = None
        best_score = 0
        
        # 训练和评估每个模型
        for name, model in models.items():
            logging.info(f"\n开始训练 {name} 模型...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            logging.info(f"{name} 交叉验证分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 在测试集上评估
            y_pred = model.predict(X_test)
            
            # 打印分类报告
            print(f"\n{name} 分类报告:")
            print(classification_report(y_test, y_pred))
            
            # 绘制混淆矩阵
            plot_confusion_matrix(y_test, y_pred, label_encoder, name)
            
            # 保存模型
            joblib.dump(model, f'models/{name.lower()}_model.pkl')
            
            # 更新最佳模型
            score = cv_scores.mean()
            if score > best_score:
                best_score = score
                best_model = (name, model)
            
            # 如果是随机森林，绘制特征重要性
            if name == 'RandomForest':
                plot_feature_importance(model, model.feature_importances_)
        
        # 记录最佳模型
        logging.info(f"\n最佳模型是 {best_model[0]}，交叉验证分数: {best_score:.4f}")
        
        return best_model[1]
    except Exception as e:
        logging.error(f"模型训练和评估失败: {str(e)}")
        return None

def main():
    # 加载数据
    features, labels = load_data()
    if features is None or labels is None:
        return
    
    # 数据预处理
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(features, labels)
    if X_train is None:
        return
    
    # 训练和评估模型
    best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder)
    if best_model is None:
        return
    
    logging.info("训练流程完成")

if __name__ == "__main__":
    main()