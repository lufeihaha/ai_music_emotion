import os
import json
import subprocess
import shutil
from tqdm import tqdm

def main():
    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 创建必要的目录
    raw_dir = config['dataset']['path']
    os.makedirs(raw_dir, exist_ok=True)
    
    print("开始下载 GTZAN 数据集...")
    
    # 使用kaggle命令行工具下载数据集
    try:
        # 下载数据集
        subprocess.run([
            'kaggle', 'datasets', 'download',
            'andradaolteanu/gtzan-dataset-music-genre-classification',
            '--path', raw_dir,
            '--unzip'
        ], check=True)
        
        print("数据集下载完成!")
        
        # 移动文件到正确的位置
        genres_dir = os.path.join(raw_dir, 'Data', 'genres_original')
        if os.path.exists(genres_dir):
            # 移动所有音频文件到raw目录
            print("整理文件...")
            for genre_dir in os.listdir(genres_dir):
                genre_path = os.path.join(genres_dir, genre_dir)
                if os.path.isdir(genre_path):
                    for audio_file in os.listdir(genre_path):
                        src = os.path.join(genre_path, audio_file)
                        dst = os.path.join(raw_dir, f"{genre_dir}_{audio_file}")
                        shutil.move(src, dst)
            
            # 清理临时文件
            print("清理临时文件...")
            shutil.rmtree(os.path.join(raw_dir, 'Data'))
            
        print(f"数据集准备完成! 文件保存在: {raw_dir}")
        print("注意: 接下来我们需要进行数据预处理和特征提取")
        
    except subprocess.CalledProcessError as e:
        print("下载失败。请确保您已安装并配置了Kaggle API:")
        print("1. 安装 Kaggle: pip install kaggle")
        print("2. 从 Kaggle 网站下载 API token (kaggle.json)")
        print("3. 将 kaggle.json 放在 ~/.kaggle/ 目录下")
        print("4. 在 Windows 上，通常是: C:\\Users\\<用户名>\\.kaggle\\kaggle.json")
        print("\n详细错误信息:", str(e))
        return

if __name__ == '__main__':
    main() 