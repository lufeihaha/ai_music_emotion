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
            return False
    else:
        print("❌ 未找到requirements.txt文件")
        return False
    
    return True

def check_required_modules():
    """检查必要的模块是否可用"""
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
            print(f"❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n缺少模块: {missing_modules}")
        print("正在尝试安装...")
        return install_requirements()
    
    return True

def check_data_directory():
    """检查数据目录结构"""
    print("\n检查数据目录...")
    
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # 创建目录
    data_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    # 检查原始数据
    audio_files = []
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    
    for genre in genres:
        genre_dir = raw_dir / genre
        if genre_dir.exists():
            wav_files = list(genre_dir.glob("*.wav"))
            audio_files.extend(wav_files)
            if wav_files:
                print(f"✓ {genre}: {len(wav_files)} 个音频文件")
            else:
                print(f"⚠️  {genre}: 目录存在但无音频文件")
        else:
            print(f"❌ {genre}: 目录不存在")
    
    if audio_files:
        print(f"\n✓ 总计找到 {len(audio_files)} 个音频文件")
        return True
    else:
        print(f"\n❌ 未找到任何音频文件")
        print("请确保音频文件按流派分类存放在 data/raw/ 目录下")
        print("目录结构应为:")
        print("data/raw/")
        for genre in genres:
            print(f"  ├── {genre}/")
            print(f"  │   ├── {genre}.00000.wav")
            print(f"  │   ├── {genre}.00001.wav")
            print(f"  │   └── ...")
        return False

def setup_project_structure():
    """设置项目结构"""
    print("\n设置项目结构...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "outputs/models",
        "outputs/results",
        "outputs/visualizations",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")

def run_training_pipeline():
    """运行训练流程"""
    print("\n" + "="*60)
    print("开始机器学习训练流程")
    print("="*60)
    
    try:
        # 更改到src目录
        src_dir = Path("src")
        if src_dir.exists():
            os.chdir(src_dir)
            print("切换到src目录")
        
        # 导入并运行集成训练器
        from integrated_trainer import IntegratedTrainer
        
        # 创建训练器实例
        trainer = IntegratedTrainer(config_path="../config.json")
        
        # 运行完整训练流程
        results = trainer.run_complete_training()
        
        print("\n🎉 训练流程完成!")
        return results
        
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def show_next_steps(has_data=False):
    """显示下一步操作指引"""
    print("\n" + "="*60)
    print("下一步操作指引")
    print("="*60)
    
    if not has_data:
        print("\n📁 获取数据:")
        print("1. 下载GTZAN数据集")
        print("   python src/download_dataset.py")
        print("\n2. 或者手动下载数据并放置到正确位置:")
        print("   - 下载GTZAN数据集")
        print("   - 解压到 data/raw/ 目录")
        print("   - 确保每个流派的音频文件在对应子目录中")
        print("\n3. 数据准备完成后重新运行:")
        print("   python run_training.py")
    else:
        print("\n🚀 训练已完成，您可以:")
        print("1. 查看训练结果: outputs/results/")
        print("2. 使用训练好的模型: outputs/models/")
        print("3. 运行Web应用: python src/app.py")
        print("4. 进行模型调优: python src/model_tuning.py")
        print("5. 生成可视化报告: python src/visualization.py")

def main():
    """主函数"""
    print("🎵 音乐情感分类项目启动")
    print("="*60)
    
    try:
        # 1. 检查Python版本
        check_python_version()
        
        # 2. 设置项目结构
        setup_project_structure()
        
        # 3. 检查和安装依赖
        if not check_required_modules():
            print("❌ 依赖检查失败，请手动安装依赖包")
            return
        
        # 4. 检查数据
        has_data = check_data_directory()
        
        if has_data:
            # 5. 运行训练流程
            results = run_training_pipeline()
            
            if results:
                print("\n✅ 所有步骤完成!")
            else:
                print("\n❌ 训练流程失败")
        
        # 6. 显示下一步指引
        show_next_steps(has_data)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()