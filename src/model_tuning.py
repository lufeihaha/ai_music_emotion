import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
from datetime import datetime

class ModelTuner:
    """模型超参数调优器"""
    
    def __init__(self, log_dir: str = "tuning_logs"):
        """
        初始化调优器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def tune_traditional_model(self, model_class: Any, param_grid: Dict[str, List[Any]],
                             X: np.ndarray, y: np.ndarray, cv: int = 5,
                             scoring: str = 'accuracy', n_jobs: int = -1,
                             method: str = 'grid') -> Dict[str, Any]:
        """
        调优传统机器学习模型
        
        Args:
            model_class: 模型类
            param_grid: 参数网格
            X: 特征矩阵
            y: 标签数组
            cv: 交叉验证折数
            scoring: 评分方式
            n_jobs: 并行作业数
            method: 搜索方法 ('grid' 或 'random')
            
        Returns:
            调优结果字典
        """
        model = model_class()
        
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv,
                scoring=scoring, n_jobs=n_jobs,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid, cv=cv,
                scoring=scoring, n_jobs=n_jobs,
                n_iter=100, verbose=1
            )
        
        search.fit(X, y)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
        
        # 记录结果
        self._log_results('traditional', results)
        
        return results
    
    def tune_deep_model(self, model_class: Any, param_ranges: Dict[str, Any],
                       X: np.ndarray, y: np.ndarray, n_trials: int = 100,
                       device: str = 'cuda') -> Dict[str, Any]:
        """
        使用Optuna调优深度学习模型
        
        Args:
            model_class: 模型类
            param_ranges: 参数范围
            X: 特征矩阵
            y: 标签数组
            n_trials: 试验次数
            device: 设备 ('cuda' 或 'cpu')
            
        Returns:
            调优结果字典
        """
        def objective(trial):
            # 从参数范围中采样
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, *param_range)
                    else:
                        params[param_name] = trial.suggest_float(param_name, *param_range)
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # 创建模型
            model = model_class(**params).to(device)
            
            # 准备数据
            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.LongTensor(y).to(device)
            dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(dataset, batch_size=params.get('batch_size', 32),
                                    shuffle=True)
            
            # 训练和评估
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
            
            for epoch in range(params.get('epochs', 100)):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # 计算验证集得分
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            
            return accuracy
        
        # 创建学习器
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        results = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
        
        # 记录结果
        self._log_results('deep', results)
        
        return results
    
    def _log_results(self, model_type: str, results: Dict[str, Any]):
        """记录调优结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.log_dir}/{model_type}_tuning_{timestamp}.json"
        
        # 确保可以序列化
        log_results = {
            'best_params': results['best_params'],
            'best_score': float(results['best_score']),
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(log_results, f, indent=4)
        
        self.logger.info(f"调优结果已保存到: {filename}")
    
    def plot_optimization_history(self, study: optuna.study.Study, save_path: Optional[str] = None):
        """
        绘制优化历史
        
        Args:
            study: Optuna学习器
            save_path: 图像保存路径
        """
        import plotly.graph_objects as go
        
        # 绘制优化历史
        fig = optuna.visualization.plot_optimization_history(study)
        
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()
    
    def plot_param_importances(self, study: optuna.study.Study, save_path: Optional[str] = None):
        """
        绘制参数重要性
        
        Args:
            study: Optuna学习器
            save_path: 图像保存路径
        """
        import plotly.graph_objects as go
        
        # 绘制参数重要性
        fig = optuna.visualization.plot_param_importances(study)
        
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()
    
    def suggest_best_model(self, results: Dict[str, Any], threshold: float = 0.95) -> str:
        """
        根据调优结果推荐最佳模型
        
        Args:
            results: 调优结果字典
            threshold: 性能阈值
            
        Returns:
            模型推荐建议
        """
        best_score = results['best_score']
        params = results['best_params']
        
        suggestion = f"最佳得分: {best_score:.4f}\n"
        suggestion += f"最佳参数: {json.dumps(params, indent=2)}\n\n"
        
        if best_score >= threshold:
            suggestion += "模型性能良好，建议使用当前配置。"
        else:
            suggestion += "建议：\n"
            suggestion += "1. 考虑增加训练数据\n"
            suggestion += "2. 尝试更复杂的模型架构\n"
            suggestion += "3. 进行特征工程\n"
            suggestion += "4. 扩大参数搜索范围"
        
        return suggestion 