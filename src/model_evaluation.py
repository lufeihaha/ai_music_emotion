import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import os
import sys
sys.path.append('.')  # 添加当前目录到Python路径

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_test_data():
    """
    加载测试数据
    """
    try:
        # 加载特征和标签
        X = np.load('data/features.npy')
        labels_df = pd.read_csv('data/processed/emotion_labels.csv')
        
        # 划分测试集
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, labels_df['emotion'].values, 
                                               test_size=0.2, random_state=42)
        
        return X_test, y_test
    except Exception as e:
        logging.error(f"加载测试数据时出错: {str(e)}")
        return None, None

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    绘制混淆矩阵
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        # 保存图像
        save_path = 'models/visualizations/confusion_matrix.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        logging.info(f"混淆矩阵已保存至 {save_path}")
    except Exception as e:
        logging.error(f"绘制混淆矩阵时出错: {str(e)}")

def calculate_metrics(y_true, y_pred, labels):
    """
    计算各项评估指标
    """
    try:
        # 生成分类报告
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        
        # 转换为DataFrame以便更好地展示
        df_report = pd.DataFrame(report).transpose()
        
        # 保存报告
        report_path = 'models/evaluation_report.csv'
        df_report.to_csv(report_path)
        
        logging.info(f"评估报告已保存至 {report_path}")
        
        # 打印主要指标
        logging.info("\n=== 模型评估指标 ===")
        logging.info(f"整体准确率: {report['accuracy']:.4f}")
        logging.info("\n各类别指标:")
        for label in labels:
            logging.info(f"\n{label}:")
            logging.info(f"精确率: {report[label]['precision']:.4f}")
            logging.info(f"召回率: {report[label]['recall']:.4f}")
            logging.info(f"F1分数: {report[label]['f1-score']:.4f}")
        
        return df_report
    except Exception as e:
        logging.error(f"计算评估指标时出错: {str(e)}")
        return None

def main():
    """
    主函数
    """
    try:
        # 加载模型（使用标准层）
        model = load_model('models/improved_rnn_model.keras')
        
        # 加载测试数据
        X_test, y_test = load_test_data()
        if X_test is None or y_test is None:
            return
        
        # 定义标签名称
        labels = ['calm', 'energetic', 'happy', 'melancholic']
        
        # 创建标签编码器
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        
        # 将字符串标签转换为数字
        if isinstance(y_test[0], str):
            y_test_encoded = label_encoder.transform(y_test)
            logging.info("真实标签已从字符串转换为数字")
        else:
            y_test_encoded = y_test
            logging.info("真实标签已经是数字格式")
        
        # 获取预测结果
        logging.info("开始模型预测...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        logging.info(f"预测完成，数据形状: {y_pred.shape}")
        logging.info(f"真实标签类型: {type(y_test_encoded[0])}, 预测标签类型: {type(y_pred_classes[0])}")
        logging.info(f"真实标签范围: {np.min(y_test_encoded)} - {np.max(y_test_encoded)}")
        logging.info(f"预测标签范围: {np.min(y_pred_classes)} - {np.max(y_pred_classes)}")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y_test_encoded, y_pred_classes, labels)
        
        # 计算评估指标
        metrics_report = calculate_metrics(y_test_encoded, y_pred_classes, labels)
        
        # 额外的性能分析
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test_encoded, y_pred_classes)
        precision = precision_score(y_test_encoded, y_pred_classes, average='weighted')
        recall = recall_score(y_test_encoded, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred_classes, average='weighted')
        
        logging.info(f"\n=== 综合性能指标 ===")
        logging.info(f"准确率: {accuracy:.4f}")
        logging.info(f"加权精确率: {precision:.4f}")
        logging.info(f"加权召回率: {recall:.4f}")
        logging.info(f"加权F1分数: {f1:.4f}")
        
        # 预测置信度分析
        max_probs = np.max(y_pred, axis=1)
        logging.info(f"\n=== 置信度分析 ===")
        logging.info(f"平均置信度: {np.mean(max_probs):.4f}")
        logging.info(f"置信度标准差: {np.std(max_probs):.4f}")
        logging.info(f"低置信度样本 (<0.5): {np.sum(max_probs < 0.5)}")
        logging.info(f"高置信度样本 (>0.8): {np.sum(max_probs > 0.8)}")
        
        logging.info("模型评估完成")
        
    except Exception as e:
        logging.error(f"模型评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 