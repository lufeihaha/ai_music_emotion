import os
import shutil
from pathlib import Path

def organize_dataset():
    # 数据目录
    raw_dir = Path('data/raw')
    
    # 获取所有音频文件
    audio_files = [f for f in raw_dir.glob('*.wav')]
    
    # 创建流派目录并移动文件
    for audio_file in audio_files:
        # 从文件名中提取流派
        genre = audio_file.name.split('_')[0]
        
        # 创建流派目录
        genre_dir = raw_dir / genre
        genre_dir.mkdir(exist_ok=True)
        
        # 移动文件
        shutil.move(str(audio_file), str(genre_dir / audio_file.name))
        print(f'Moved {audio_file.name} to {genre_dir}')

if __name__ == '__main__':
    print('开始整理数据集...')
    organize_dataset()
    print('数据集整理完成！') 