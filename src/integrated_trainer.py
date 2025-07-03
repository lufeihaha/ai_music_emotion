import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import joblib
except ImportError as e:
    print(f"导入scikit-learn模块时出错: {e}")
    print("请安装: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch不可用，将使用scikit-learn模型")
    TORCH_AVAILABLE = False

from data_preprocessing import DataPreprocessor
from feature_extractor import AdvancedFeatureExtractor
from data_augmentation import DataAugmentor

class DeepModel(nn.Module):
    """深度学习模型（当PyTorch可用时）"""
    
    def __init__(self, input_size: int, num_classes: int, config: Dict):
        super(DeepModel, self).__init__()
        
        self.dropout_rate = config.get('dropout_rate', 0.3)
        dense_layers = config.get('dense_layers', [256, 128])
        
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_size, dense_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        
        # 隐藏层
        for i in range(len(dense_layers) - 1):
            layers.append(nn.Linear(dense_layers[i], dense_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(dense_layers[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class IntegratedTrainer:
    """集成训练器，整合完整的机器学习流程"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化集成训练器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建输出目录
        self.output_dir = Path(self.config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # 模型保存目录
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        # 结果保存目录
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.preprocessor = None
        self.feature_extractor = None
        self.augmentor = None
        self.scaler = None
        self.models = {}
        
        # 训练历史
        self.training_history = {}
    
    def run_preprocessing(self) -> Dict:
        """
        运行数据预处理
        
        Returns:
            预处理结果
        """
        self.logger.info("开始数据预处理...")
        
        # 创建数据预处理器
        self.preprocessor = DataPreprocessor(config_path="../config.json")
        
        # 运行预处理
        result = self.preprocessor.run_preprocessing()
        
        # 保存预处理结果
        with open(self.results_dir / 'preprocessing_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        self.logger.info("数据预处理完成")
        return result
    
    def run_feature_extraction(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        运行特征提取
        
        Returns:
            特征矩阵、情感标签、标签ID
        """
        self.logger.info("开始特征提取...")
        
        # 创建特征提取器
        self.feature_extractor = AdvancedFeatureExtractor(config_path="../config.json")
        
        # 读取预处理后的片段数据
        segments_path = Path(self.config['dataset']['processed_path']) / "segments_catalog.csv"
        
        if not segments_path.exists():
            raise FileNotFoundError(f"未找到片段数据文件: {segments_path}")
        
        segments_df = pd.read_csv(segments_path)
        
        # 批量提取特征
        X, emotions, label_ids = self.feature_extractor.extract_features_batch(segments_df)
        
        # 保存特征
        features_path = Path(self.config['dataset']['processed_path']) / "features.csv"
        self.feature_extractor.save_features(X, emotions, label_ids, str(features_path))
        
        self.logger.info(f"特征提取完成，特征矩阵形状: {X.shape}")
        return X, emotions, label_ids
    
    def prepare_training_data(self, X: np.ndarray, emotions: np.ndarray, label_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            X: 特征矩阵
            emotions: 情感标签
            label_ids: 标签ID
            
        Returns:
            训练特征、测试特征、训练标签、测试标签
        """
        self.logger.info("准备训练数据...")
        
        # 过滤有效数据
        valid_mask = label_ids != -1
        X_valid = X[valid_mask]
        emotions_valid = emotions[valid_mask]
        label_ids_valid = label_ids[valid_mask]
        
        # 数据增强
        if self.config['data_augmentation']['enabled']:
            self.logger.info("开始数据增强...")
            self.augmentor = DataAugmentor(sr=self.config['dataset']['sample_rate'])
            
            # 统计每个类别的样本数
            unique_labels, counts = np.unique(label_ids_valid, return_counts=True)
            max_count = max(counts)
            
            # 计算每个类别需要增强的数量
            augmentations_per_class = {}
            for label, count in zip(unique_labels, counts):
                emotion = emotions_valid[label_ids_valid == label][0]
                augmentations_per_class[emotion] = max_count - count
            
            # 进行增强（这里简化为复制样本，实际应该对音频进行增强）
            X_augmented_list = [X_valid]
            emotions_augmented_list = [emotions_valid]
            label_ids_augmented_list = [label_ids_valid]
            
            for label, aug_count in augmentations_per_class.items():
                if aug_count > 0:
                    label_mask = emotions_valid == label
                    label_data = X_valid[label_mask]
                    
                    # 随机重复样本进行简单增强
                    if len(label_data) > 0:
                        aug_indices = np.random.choice(len(label_data), aug_count, replace=True)
                        X_aug = label_data[aug_indices]
                        emotions_aug = np.full(aug_count, label)
                        label_ids_aug = np.full(aug_count, label_ids_valid[emotions_valid == label][0])
                        
                        X_augmented_list.append(X_aug)
                        emotions_augmented_list.append(emotions_aug)
                        label_ids_augmented_list.append(label_ids_aug)
            
            X_final = np.vstack(X_augmented_list)
            emotions_final = np.concatenate(emotions_augmented_list)
            label_ids_final = np.concatenate(label_ids_augmented_list)
            
            self.logger.info(f"数据增强完成，样本数从 {len(X_valid)} 增加到 {len(X_final)}")
        else:
            X_final = X_valid
            emotions_final = emotions_valid
            label_ids_final = label_ids_valid
        
        # 数据标准化
        self.logger.info("数据标准化...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_final)
        
        # 保存标准化器
        scaler_path = self.models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # 分割训练集和测试集
        training_config = self.config['training']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            label_ids_final,
            test_size=training_config['test_size'],
            random_state=training_config['random_state'],
            stratify=label_ids_final if training_config['stratify'] else None
        )
        
        self.logger.info(f"数据准备完成:")
        self.logger.info(f"  训练集: {X_train.shape}")
        self.logger.info(f"  测试集: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_sklearn_models(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        训练scikit-learn模型
        
        Args:
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练标签
            y_test: 测试标签
            
        Returns:
            模型评估结果
        """
        self.logger.info("开始训练scikit-learn模型...")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"训练 {model_name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            
            # 分类报告
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # 保存模型
            model_path = self.models_dir / f'{model_name}.pkl'
            joblib.dump(model, model_path)
            
            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'model_path': str(model_path)
            }
            
            self.models[model_name] = model
            
            self.logger.info(f"{model_name} 训练完成，准确率: {accuracy:.4f}")
        
        return results
    
    def train_deep_model(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        训练深度学习模型
        
        Args:
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练标签
            y_test: 测试标签
            
        Returns:
            模型评估结果
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch不可用，跳过深度学习模型训练")
            return {}
        
        self.logger.info("开始训练深度学习模型...")
        
        # 模型配置
        model_config = self.config['model']['params']
        num_classes = len(np.unique(y_train))
        input_size = X_train.shape[1]
        
        # 创建模型
        model = DeepModel(input_size, num_classes, model_config)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
        
        # 训练循环
        model.train()
        training_history = {'loss': [], 'accuracy': []}
        
        for epoch in range(model_config['epochs']):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_accuracy = correct / total
            training_history['loss'].append(epoch_loss / len(train_loader))
            training_history['accuracy'].append(epoch_accuracy)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{model_config["epochs"]}], Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.4f}')
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = (test_predicted == y_test_tensor).float().mean().item()
        
        # 保存模型
        model_path = self.models_dir / 'deep_model.pth'
        torch.save(model.state_dict(), model_path)
        
        self.models['deep_model'] = model
        
        result = {
            'accuracy': test_accuracy,
            'training_history': training_history,
            'model_path': str(model_path)
        }
        
        self.logger.info(f"深度学习模型训练完成，测试准确率: {test_accuracy:.4f}")
        
        return {'deep_model': result}
    
    def load_preprocessed_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载预处理的特征数据（用于模拟数据或预生成的特征）
        
        Returns:
            特征矩阵、情感标签、标签ID
        """
        self.logger.info("加载预处理的特征数据...")
        
        # 尝试多个可能的路径
        possible_paths = [
            Path(self.config['dataset']['processed_path']) / "features.csv",
            Path("../data/processed/features.csv"),
            Path("data/processed/features.csv"),
            Path("/workspace/data/processed/features.csv")
        ]
        
        features_path = None
        for path in possible_paths:
            self.logger.info(f"尝试路径: {path}, 存在: {path.exists()}")
            if path.exists():
                features_path = path
                break
        
        if not features_path:
            raise FileNotFoundError(f"未找到特征文件，尝试的路径: {[str(p) for p in possible_paths]}")
        
        self.logger.info(f"使用特征文件: {features_path}")
        
        # 读取特征数据
        features_df = pd.read_csv(features_path)
        
        # 分离特征和标签
        feature_columns = [col for col in features_df.columns if col not in ['emotion', 'label_id']]
        X = features_df[feature_columns].values
        emotions = features_df['emotion'].values
        label_ids = features_df['label_id'].values
        
        self.logger.info(f"加载特征数据完成，特征矩阵形状: {X.shape}")
        self.logger.info(f"情感类别: {np.unique(emotions)}")
        
        return X, emotions, label_ids

    def run_mock_training(self) -> Dict:
        """
        使用模拟数据运行训练流程（跳过音频处理）
        
        Returns:
            完整的训练结果
        """
        start_time = datetime.now()
        self.logger.info("开始模拟数据机器学习训练流程...")
        
        # 1. 加载预处理的特征数据
        X, emotions, label_ids = self.load_preprocessed_features()
        
        # 2. 准备训练数据
        X_train, X_test, y_train, y_test = self.prepare_training_data(X, emotions, label_ids)
        
        # 3. 训练传统机器学习模型
        sklearn_results = self.train_sklearn_models(X_train, X_test, y_train, y_test)
        
        # 4. 训练深度学习模型
        deep_results = self.train_deep_model(X_train, X_test, y_train, y_test)
        
        # 5. 汇总结果
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        final_results = {
            'preprocessing': {
                'total_files': len(emotions),
                'total_segments': len(emotions),
                'train_samples': len(X_train),
                'val_samples': 0,  # 模拟数据中没有单独的验证集
                'test_samples': len(X_test),
                'emotions': list(np.unique(emotions[emotions != 'unknown'])) if 'unknown' in emotions else list(np.unique(emotions))
            },
            'feature_extraction': {
                'feature_shape': X.shape,
                'num_emotions': len(np.unique(emotions[emotions != 'unknown'])) if 'unknown' in emotions else len(np.unique(emotions)),
                'emotions': list(np.unique(emotions[emotions != 'unknown'])) if 'unknown' in emotions else list(np.unique(emotions))
            },
            'sklearn_models': sklearn_results,
            'deep_models': deep_results,
            'training_time_seconds': training_time,
            'timestamp': start_time.isoformat(),
            'data_type': 'mock'  # 标识这是模拟数据
        }
        
        # 保存完整结果
        results_path = self.results_dir / 'complete_training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为JSON可序列化类型
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj
            
            json.dump(final_results, f, ensure_ascii=False, indent=2, default=convert_numpy)
        
        # 打印摘要
        self.print_training_summary(final_results)
        
        self.logger.info(f"模拟数据训练流程完成，总耗时: {training_time:.2f} 秒")
        
        return final_results
    
    def run_complete_training(self) -> Dict:
        """
        运行完整的训练流程
        
        Returns:
            完整的训练结果
        """
        start_time = datetime.now()
        self.logger.info("开始完整的机器学习训练流程...")
        
        # 1. 数据预处理
        preprocessing_result = self.run_preprocessing()
        
        # 2. 特征提取
        X, emotions, label_ids = self.run_feature_extraction()
        
        # 3. 准备训练数据
        X_train, X_test, y_train, y_test = self.prepare_training_data(X, emotions, label_ids)
        
        # 4. 训练传统机器学习模型
        sklearn_results = self.train_sklearn_models(X_train, X_test, y_train, y_test)
        
        # 5. 训练深度学习模型
        deep_results = self.train_deep_model(X_train, X_test, y_train, y_test)
        
        # 6. 汇总结果
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        final_results = {
            'preprocessing': preprocessing_result,
            'feature_extraction': {
                'feature_shape': X.shape,
                'num_emotions': len(np.unique(emotions[emotions != 'unknown'])),
                'emotions': list(np.unique(emotions[emotions != 'unknown']))
            },
            'sklearn_models': sklearn_results,
            'deep_models': deep_results,
            'training_time_seconds': training_time,
            'timestamp': start_time.isoformat()
        }
        
        # 保存完整结果
        results_path = self.results_dir / 'complete_training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为JSON可序列化类型
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float32):
                    return float(obj)
                elif isinstance(obj, np.int64):
                    return int(obj)
                return obj
            
            import json
            json.dump(final_results, f, ensure_ascii=False, indent=2, default=convert_numpy)
        
        # 打印摘要
        self.print_training_summary(final_results)
        
        self.logger.info(f"完整训练流程完成，总耗时: {training_time:.2f} 秒")
        
        return final_results
    
    def print_training_summary(self, results: Dict):
        """
        打印训练摘要
        
        Args:
            results: 训练结果字典
        """
        print("\n" + "="*60)
        print("           机器学习训练完成摘要")
        print("="*60)
        
        # 数据摘要
        print("\n📊 数据摘要:")
        print(f"  总文件数: {results['preprocessing']['total_files']}")
        print(f"  总片段数: {results['preprocessing']['total_segments']}")
        print(f"  训练样本: {results['preprocessing']['train_samples']}")
        print(f"  验证样本: {results['preprocessing']['val_samples']}")
        print(f"  测试样本: {results['preprocessing']['test_samples']}")
        print(f"  情感类别: {results['feature_extraction']['emotions']}")
        
        # 特征摘要
        print(f"\n🔧 特征提取:")
        print(f"  特征维度: {results['feature_extraction']['feature_shape']}")
        print(f"  情感类别数: {results['feature_extraction']['num_emotions']}")
        
        # 模型性能
        print(f"\n🎯 模型性能:")
        for model_name, model_result in results['sklearn_models'].items():
            print(f"  {model_name}: {model_result['accuracy']:.4f}")
        
        if results['deep_models']:
            for model_name, model_result in results['deep_models'].items():
                print(f"  {model_name}: {model_result['accuracy']:.4f}")
        
        # 找出最佳模型
        all_accuracies = {}
        all_accuracies.update({name: result['accuracy'] for name, result in results['sklearn_models'].items()})
        all_accuracies.update({name: result['accuracy'] for name, result in results['deep_models'].items()})
        
        if all_accuracies:
            best_model = max(all_accuracies, key=all_accuracies.get)
            best_accuracy = all_accuracies[best_model]
            print(f"\n🏆 最佳模型: {best_model} (准确率: {best_accuracy:.4f})")
        
        # 训练时间
        print(f"\n⏱️  总训练时间: {results['training_time_seconds']:.2f} 秒")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    try:
        # 创建集成训练器
        trainer = IntegratedTrainer()
        
        # 运行完整训练流程
        results = trainer.run_complete_training()
        
        return results
        
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()