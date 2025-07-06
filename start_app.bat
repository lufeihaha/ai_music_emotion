@echo off
echo 🎵 音乐情感识别系统启动脚本
echo ================================

echo 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python未安装或不在PATH中
    pause
    exit /b 1
)

echo 安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)

echo ✅ 环境准备完成
echo 🚀 启动Web应用...
echo 请在浏览器中访问: http://localhost:5000
echo ================================

python run_app.py

pause 