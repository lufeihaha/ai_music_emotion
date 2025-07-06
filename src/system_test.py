"""
系统集成与功能测试脚本
"""
import os
import subprocess

def test_model_loading():
    try:
        import tensorflow as tf
        model_path = 'models/improved_hybrid_model.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("模型加载成功")
        else:
            print("未找到模型文件")
    except Exception as e:
        print("模型加载失败：", e)

def test_web_app():
    app_path = 'src/app.py'
    if os.path.exists(app_path):
        try:
            result = subprocess.run(['python', app_path, '--help'], capture_output=True, text=True, timeout=10)
            print("Web系统入口检测：", result.stdout[:200])
        except Exception as e:
            print("Web系统检测失败：", e)
    else:
        print("未检测到Web系统入口文件")

def test_predict_script():
    predict_path = 'src/predict.py'
    if os.path.exists(predict_path):
        try:
            result = subprocess.run(['python', predict_path, '--help'], capture_output=True, text=True, timeout=10)
            print("预测脚本检测：", result.stdout[:200])
        except Exception as e:
            print("预测脚本检测失败：", e)
    else:
        print("未检测到预测脚本")

if __name__ == "__main__":
    test_model_loading()
    test_web_app()
    test_predict_script()
