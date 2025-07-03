# AI歌声情绪识别与可视化系统

这是一个基于深度学习的歌声情绪识别系统，能够分析音频中的情感特征并进行可视化展示。

## 功能特点

- 音频上传与处理
- 实时情绪识别
- 情感特征可视化
- 支持多种音频格式

## 系统要求

- Python 3.8+
- 相关依赖包（见requirements.txt）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [repository-url]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行应用：
```bash
python src/app.py
```

## 项目结构

```
.
├── data/               # 数据目录
├── models/            # 模型存储
├── src/               # 源代码
├── static/            # 静态文件
├── templates/         # HTML模板
└── requirements.txt   # 项目依赖
```

## 使用说明

1. 启动服务器后访问 http://localhost:5000
2. 上传音频文件
3. 系统将自动分析并显示情绪变化曲线

## 技术栈

- 后端：Python, Flask
- 音频处理：Librosa
- 深度学习：TensorFlow
- 可视化：ECharts
- 前端：HTML5, JavaScript 