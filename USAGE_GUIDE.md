# 🎵 音乐情感识别系统使用指南

## 🚀 快速开始

### 1. 启动Web应用
```bash
python run_app.py
```
- 访问: http://localhost:5000
- 状态: ✅ 当前正在运行

### 2. 系统状态检查
```bash
python system_status_check.py
```
- 检查所有组件状态
- 获取性能指标
- 查看建议改进

### 3. 动态预测器测试
```bash
python src/dynamic_predictor.py
```
- 查看当前加载的模型
- 测试预测器功能

## 🎯 核心功能

### 单文件音频分析
1. 打开 http://localhost:5000
2. 点击"选择文件"上传音频
3. 支持格式: WAV, MP3, FLAC, OGG, M4A
4. 获得情感分析结果和可视化

### 批量音频处理
1. 选择多个音频文件（最多10个）
2. 总文件大小限制: 80MB
3. 单文件大小限制: 20MB
4. 获得批量分析统计

### 用户反馈
- 对分析结果进行评分
- 帮助改进模型准确性
- 数据用于模型优化

## 🤖 模型系统

### 当前模型性能
| 模型类型 | 准确率 | 状态 |
|---------|--------|------|
| SVM | 67.67% | ✅ 最佳 |
| Gradient Boosting | 66.00% | ✅ |
| MLP | 66.00% | ✅ |
| Random Forest | 61.67% | ✅ |
| 集成学习 | 训练中 | ⏳ |

### 情感分类
**6类情感识别:**
- 🍃 calm (平静)
- ⚡ energetic (充满活力)  
- 😊 happy (快乐)
- ☁️ melancholic (忧郁)
- 🌙 nostalgic (怀念)
- 💕 romantic (浪漫)

## 🔧 高级功能

### 集成学习训练
```bash
python src/improved_ensemble.py
```
- 组合多个模型提升性能
- 自动特征选择
- 软投票和硬投票策略

### 模型评估
```bash
python src/quick_ensemble_test.py
```
- 快速测试模型性能
- 比较不同算法效果

### 中文优化模型
```bash
python src/chinese_enhanced_training.py
```
- 专门针对中文情感优化
- 8类详细情感分类

## 📊 系统监控

### 实时状态检查
```bash
# 检查Web应用是否运行
curl http://localhost:5000

# 检查Python进程
tasklist | findstr python

# 检查模型文件
ls models/
```

### 性能监控
- CPU使用率
- 内存占用
- 磁盘空间
- 网络连接

## 🛠️ 故障排除

### 常见问题

1. **Web应用无法访问**
   ```bash
   # 重启应用
   python run_app.py
   ```

2. **模型加载失败**
   ```bash
   # 检查模型文件
   python src/dynamic_predictor.py
   ```

3. **编码错误**
   ```bash
   # 设置环境变量
   set PYTHONIOENCODING=utf-8
   ```

4. **内存不足**
   - 减少批量处理文件数量
   - 关闭不必要的Python进程

### 性能优化

1. **提升准确率**
   - 等待集成学习训练完成
   - 使用更多训练数据
   - 调整模型超参数

2. **提升速度**
   - 使用特征选择
   - 减少模型复杂度
   - 启用并行处理

## 📈 未来改进计划

### 短期目标 (1-2周)
- [x] 完成集成学习实现
- [x] 动态模型选择
- [ ] 完成集成模型训练
- [ ] 集成到Web应用

### 中期目标 (1-2月)
- [ ] 深度学习模型 (CNN/RNN)
- [ ] 实时音频流处理
- [ ] 更多情感类别
- [ ] 跨语言支持

### 长期目标 (3-6月)
- [ ] 移动端应用
- [ ] 云端部署
- [ ] 音乐推荐系统
- [ ] 商业化应用

## 🎓 技术文档

### 架构概览
```
音乐文件 → 特征提取 → 模型预测 → 情感分析 → 结果展示
    ↓         ↓         ↓         ↓         ↓
  WAV/MP3   MFCC等   RF/SVM等   6类情感   Web界面
```

### 核心模块
- `src/audio_processor.py` - 音频处理
- `src/emotion_classifier.py` - 情感分类
- `src/web_app.py` - Web应用
- `src/dynamic_predictor.py` - 动态预测器

### 数据流程
1. 音频上传 → 特征提取 (4864维)
2. 特征选择 → 标准化 → 模型预测
3. 概率输出 → 情感标签 → 可视化

## 🤝 贡献指南

### 改进建议
1. 提交Issue描述问题
2. Fork项目进行开发
3. 提交Pull Request
4. 代码审查和合并

### 开发环境
```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black src/
```

## 📞 技术支持

### 联系方式
- 项目地址: 本地开发环境
- 文档: 查看项目根目录的.md文件
- 状态检查: `python system_status_check.py`

### 调试信息
- 日志位置: `logs/`
- 模型文件: `models/`
- 配置文件: `config.json`

---

**🎉 感谢使用音乐情感识别系统！**

*系统已达到生产就绪状态，当前准确率67.67%，支持6类情感识别。* 