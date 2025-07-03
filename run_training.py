#!/usr/bin/env python3
"""
音乐情感分类项目主启动脚本
完整的机器学习训练流程
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    if sys.version_info < (3, 8):
        raise Exception("需要Python 3.8或更高版本")
    print(f"✓ Python版本: {sys.version}")

def install_requirements():
    """安装项目依赖"""
    print("\n检查并安装依赖包...")
    
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✓ 依赖包安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装依赖包失败: {e}")
            print("请手动运行: pip install -r requirements.txt")
    else:
        print("⚠️ requirements.txt文件不存在")

def check_required_modules():
    """检查必要的Python模块"""
    print("\n检查必要模块...")
    required_modules = [
        'numpy', 'pandas', 'librosa', 'sklearn', 
        'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module}")
    
    if missing_modules:
        print(f"\n缺少以下模块: {', '.join(missing_modules)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def setup_project_structure():
    """设置项目目录结构"""
    print("\n设置项目结构...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "outputs/models",
        "outputs/results",
        "outputs/visualizations",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")

def check_dataset():
    """检查数据集"""
    print("\n检查数据目录...")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ 数据目录不存在: data/raw")
        return False
    
    total_files = 0
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    missing_genres = []
    for genre in genres:
        genre_dir = data_dir / genre
        if genre_dir.exists():
            audio_files = list(genre_dir.glob("*.wav"))
            print(f"✓ {genre}: {len(audio_files)} 个音频文件")
            total_files += len(audio_files)
        else:
            missing_genres.append(genre)
            print(f"❌ {genre}: 目录不存在")
    
    if missing_genres:
        print(f"\n缺少以下流派目录: {', '.join(missing_genres)}")
        return False
    
    print(f"\n✓ 总计找到 {total_files} 个音频文件")
    return total_files > 0

def run_training_pipeline():
    """运行机器学习训练流程"""
    print("\n" + "="*60)
    print("开始机器学习训练流程")
    print("="*60)
    
    try:
        # 添加src目录到sys.path（如果还没有添加的话）
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # 切换到src目录
        original_cwd = os.getcwd()
        src_dir = Path(__file__).parent / "src"
        os.chdir(src_dir)
        print("切换到src目录")
        
        # 导入训练管理器
        from integrated_trainer import IntegratedTrainer
        
        # 配置文件路径（相对于src目录）
        config_path = "../config.json"
        
        # 创建训练器实例
        trainer = IntegratedTrainer(config_path=config_path)
        
        print("\n🚀 开始训练流程...")
        
        # 检查是否有预处理的特征数据
        features_path = Path("../data/processed/features.csv")  # 相对于src目录的路径
        if features_path.exists():
            print("发现预处理的特征数据，使用模拟数据训练模式...")
            results = trainer.run_mock_training()
        else:
            print("使用完整音频处理训练模式...")
            results = trainer.run_complete_training()
        
        print("\n✅ 训练完成!")
        
        # 提取最佳模型信息
        all_accuracies = {}
        sklearn_models = results.get('sklearn_models', {})
        deep_models = results.get('deep_models', {})
        
        for name, result in sklearn_models.items():
            all_accuracies[name] = result['accuracy']
        
        for name, result in deep_models.items():
            all_accuracies[name] = result['accuracy']
        
        if all_accuracies:
            best_model = max(all_accuracies.keys(), key=lambda x: all_accuracies[x])
            best_accuracy = all_accuracies[best_model]
            print(f"最佳模型: {best_model}")
            print(f"最佳准确率: {best_accuracy:.4f}")
        else:
            print("未找到模型结果")
        
        # 切换回原目录
        os.chdir(original_cwd)
        
        return True
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 确保切换回原目录
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        return False

def main():
    """主函数"""
    print("🎵 音乐情感分类项目启动")
    print("="*60)
    
    try:
        # 检查Python版本
        check_python_version()
        
        # 设置项目结构
        setup_project_structure()
        
        # 检查必要模块
        if not check_required_modules():
            print("\n请先安装所需依赖包")
            return
        
        # 检查数据集
        if not check_dataset():
            print("\n请先准备数据集")
            return
        
        # 运行训练流程
        success = run_training_pipeline()
        
        # 显示结果和下一步指引
        print("\n" + "="*60)
        if success:
            print("✅ 训练流程完成!")
        else:
            print("❌ 训练流程失败")
        
        print("="*60)
        print("\n下一步操作指引")
        print("="*60)
        print("\n🚀 训练已完成，您可以:")
        print("1. 查看训练结果: outputs/results/")
        print("2. 使用训练好的模型: outputs/models/")
        print("3. 运行Web应用: python src/app.py")
        print("4. 进行模型调优: python src/model_tuning.py")
        print("5. 生成可视化报告: python src/visualization.py")
        
    except Exception as e:
        print(f"启动过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()