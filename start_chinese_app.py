#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动中文音乐情感识别Web应用
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    print("🎵 中文音乐情感识别系统")
    print("=" * 50)
    
    # 检查中文模型是否存在
    chinese_model_path = project_root / "models" / "chinese_emotion"
    if not chinese_model_path.exists():
        print("❌ 中文情感模型不存在")
        print("💡 正在训练中文情感模型...")
        
        # 训练中文模型
        try:
            from src.chinese_emotion_classifier import main as train_model
            train_model()
            print("✅ 中文情感模型训练完成")
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            return
    else:
        print("✅ 中文情感模型已存在")
    
    # 启动Web应用
    print("🚀 启动中文情感识别Web应用...")
    
    try:
        from src.chinese_web_app import main as start_app
        start_app()
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        print("💡 请检查依赖是否安装完整")

if __name__ == "__main__":
    main() 