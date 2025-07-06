#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 确保在正确的目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(project_root, 'src'))

print(f"🔍 项目根目录: {project_root}")
print(f"🔍 当前工作目录: {os.getcwd()}")
print(f"🔍 模板目录存在: {os.path.exists('templates')}")
print(f"🔍 music_emotion_app.html存在: {os.path.exists('templates/music_emotion_app.html')}")

# 导入并运行web应用
from src.web_app import app, predictor

if __name__ == '__main__':
    print("🎵 启动音乐情感识别Web应用...")
    print("📊 模型状态:", "✅ 已加载" if predictor is not None else "❌ 未加载")
    print("🌐 访问地址: http://localhost:5000")
    print("🔧 调试模式: 已启用")
    
    # 设置模板文件夹为当前目录下的templates
    app.template_folder = os.path.join(project_root, 'templates')
    print(f"🔍 模板文件夹设置为: {app.template_folder}")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 