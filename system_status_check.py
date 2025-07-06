#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音乐情感识别系统状态检查
"""

import os
import sys
import requests
import subprocess
import time
from datetime import datetime

def check_web_app_status():
    """检查Web应用状态"""
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            return "✅ 正常运行"
        else:
            return f"⚠️ 异常状态 (状态码: {response.status_code})"
    except requests.exceptions.ConnectionError:
        return "❌ 未运行"
    except requests.exceptions.Timeout:
        return "⏳ 响应超时"
    except Exception as e:
        return f"❌ 错误: {e}"

def check_model_files():
    """检查模型文件"""
    model_paths = [
        ('models/improved_ensemble', '改进集成模型'),
        ('models/chinese_optimized', '中文优化模型'),
        ('models/unified_results', '统一训练模型'),
        ('models', '基础模型')
    ]
    
    available_models = []
    for path, description in model_paths:
        if os.path.exists(path):
            files = os.listdir(path) if os.path.isdir(path) else [os.path.basename(path)]
            pkl_files = [f for f in files if f.endswith('.pkl')]
            if pkl_files:
                available_models.append(f"✅ {description} ({len(pkl_files)} 文件)")
            else:
                available_models.append(f"⚠️ {description} (无.pkl文件)")
        else:
            available_models.append(f"❌ {description} (不存在)")
    
    return available_models

def check_data_integrity():
    """检查数据完整性"""
    data_status = {}
    
    # 检查特征文件
    if os.path.exists('data/features.npy'):
        try:
            import numpy as np
            features = np.load('data/features.npy')
            data_status['features'] = f"✅ {features.shape}"
        except Exception as e:
            data_status['features'] = f"❌ 加载失败: {e}"
    else:
        data_status['features'] = "❌ 不存在"
    
    # 检查标签文件
    if os.path.exists('data/processed/emotion_labels.csv'):
        try:
            import pandas as pd
            labels = pd.read_csv('data/processed/emotion_labels.csv')
            data_status['labels'] = f"✅ {len(labels)} 样本"
        except Exception as e:
            data_status['labels'] = f"❌ 加载失败: {e}"
    else:
        data_status['labels'] = "❌ 不存在"
    
    # 检查原始音频文件
    raw_dirs = ['data/raw', 'data/processed']
    for dir_name in raw_dirs:
        if os.path.exists(dir_name):
            try:
                total_files = sum([len(files) for r, d, files in os.walk(dir_name) if files])
                data_status[dir_name] = f"✅ {total_files} 文件"
            except Exception as e:
                data_status[dir_name] = f"❌ 统计失败: {e}"
        else:
            data_status[dir_name] = "❌ 不存在"
    
    return data_status

def check_python_processes():
    """检查Python进程"""
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                python_processes = [line for line in lines if 'python.exe' in line]
                return f"✅ {len(python_processes)} 个Python进程"
            else:
                return "❌ 无法检查进程"
        else:  # Unix/Linux
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if result.returncode == 0:
                python_processes = [line for line in result.stdout.split('\n') if 'python' in line]
                return f"✅ {len(python_processes)} 个Python进程"
            else:
                return "❌ 无法检查进程"
    except Exception as e:
        return f"❌ 进程检查失败: {e}"

def check_system_resources():
    """检查系统资源"""
    try:
        import psutil
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        
        return {
            'cpu': f"CPU: {cpu_percent:.1f}%",
            'memory': f"内存: {memory_percent:.1f}%",
            'disk': f"磁盘: {disk_percent:.1f}%"
        }
    except ImportError:
        return {'resources': '❌ 需要安装psutil库'}
    except Exception as e:
        return {'resources': f'❌ 资源检查失败: {e}'}

def main():
    """主函数 - 系统状态检查"""
    print("🎵 音乐情感识别系统状态检查")
    print("=" * 60)
    print(f"⏰ 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Web应用状态
    print("🌐 Web应用状态:")
    web_status = check_web_app_status()
    print(f"  http://localhost:5000 - {web_status}")
    print()
    
    # 2. 模型文件状态
    print("🤖 模型文件状态:")
    model_status = check_model_files()
    for status in model_status:
        print(f"  {status}")
    print()
    
    # 3. 数据完整性
    print("📊 数据完整性:")
    data_status = check_data_integrity()
    for key, status in data_status.items():
        print(f"  {key}: {status}")
    print()
    
    # 4. Python进程
    print("🔄 Python进程:")
    process_status = check_python_processes()
    print(f"  {process_status}")
    print()
    
    # 5. 系统资源
    print("💻 系统资源:")
    resource_status = check_system_resources()
    for key, status in resource_status.items():
        print(f"  {status}")
    print()
    
    # 6. 动态预测器状态
    print("🔍 动态预测器状态:")
    try:
        sys.path.append('src')
        from dynamic_predictor import get_model_status
        predictor_status = get_model_status()
        if predictor_status['status'] == 'loaded':
            model_info = predictor_status['model_info']
            print(f"  ✅ 模型已加载: {model_info['description']}")
            print(f"  📊 类型: {model_info['type']}")
            print(f"  🎭 情感类别: {len(predictor_status['emotion_classes'])} 类")
        else:
            print(f"  ❌ 模型加载失败: {predictor_status['error']}")
    except Exception as e:
        print(f"  ❌ 预测器检查失败: {e}")
    print()
    
    # 7. 建议
    print("💡 系统建议:")
    if "正常运行" in web_status:
        print("  ✅ Web应用运行正常，可以进行音乐情感分析")
    else:
        print("  ⚠️ 建议重启Web应用: python run_app.py")
    
    if any("改进集成模型" in status and "✅" in status for status in model_status):
        print("  ✅ 集成模型可用，系统性能最优")
    else:
        print("  💡 建议完成集成学习训练以提升性能")
    
    print("\n🎉 系统状态检查完成!")

if __name__ == "__main__":
    main() 