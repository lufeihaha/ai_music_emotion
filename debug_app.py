#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 确保在正确的目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

print(f"🔍 项目根目录: {project_root}")
print(f"🔍 当前工作目录: {os.getcwd()}")

# 检查关键文件
print("\n📁 检查关键文件:")
files_to_check = [
    'templates/music_emotion_app.html',
    'models/randomforest_model.pkl',
    'models/scaler.pkl', 
    'models/label_encoder.pkl',
    'models/unified_results/randomforest_model.pkl',
    'models/unified_results/scaler.pkl',
    'models/unified_results/label_encoder.pkl'
]

for file_path in files_to_check:
    exists = os.path.exists(file_path)
    print(f"  {'✅' if exists else '❌'} {file_path}")

print("\n🧪 测试模块导入:")
try:
    from src.web_app import app, predictor
    print("✅ Web应用模块导入成功")
    print(f"✅ 预测器状态: {'已加载' if predictor is not None else '未加载'}")
    
    # 测试模型组件
    if predictor:
        print(f"✅ 模型类型: {type(predictor.model)}")
        print(f"✅ 标准化器类型: {type(predictor.scaler)}")
        print(f"✅ 标签编码器类型: {type(predictor.label_encoder)}")
        if hasattr(predictor.label_encoder, 'classes_'):
            print(f"✅ 情感类别: {predictor.label_encoder.classes_}")
    
except Exception as e:
    print(f"❌ 导入失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == '__main__':
    print("\n🚀 启动调试Web应用...")
    print("🌐 访问地址: http://localhost:5000")
    print("🔧 调试模式已启用")
    
    # 设置模板文件夹
    app.template_folder = os.path.join(project_root, 'templates')
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        import traceback
        traceback.print_exc() 