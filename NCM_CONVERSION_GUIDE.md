# 🎵 网易云音乐NCM格式转换指南

## 问题说明

网易云音乐下载的文件是 **`.ncm`** 格式，这是网易云音乐专有的加密格式：
- **只能在网易云音乐应用中播放**
- **使用AES+RC4加密算法保护**
- **我们的情感分析系统无法直接读取**

## 解决方案

我们为你提供了专门的转换工具来处理这个问题！

### 方法一：使用我们的转换工具（推荐）

```bash
# 1. 确保已安装依赖
pip install pycryptodome

# 2. 转换单个文件
python src/ncm_converter.py /path/to/your/song.ncm

# 3. 转换整个目录
python src/ncm_converter.py /path/to/music/folder/
```

### 方法二：使用第三方工具

#### 在线工具（谨慎使用）
- **注意**：不建议使用在线工具上传版权音乐
- 可能存在隐私和版权风险

#### 桌面工具
- **ncmdump** (GitHub开源项目)
- **NCM2MP3** (Java版本)
- **WyMusicConvert** (Windows命令行工具)

## 使用步骤详解

### 第一步：找到你的NCM文件
网易云音乐默认下载位置：
- **Windows**: `C:\Users\用户名\Music\CloudMusic\`
- **macOS**: `~/Music/CloudMusic/`
- **Linux**: `~/Music/CloudMusic/`

### 第二步：运行转换工具
```bash
# 进入项目目录
cd /workspace

# 转换示例（假设你的文件在Downloads目录）
python src/ncm_converter.py ~/Downloads/song.ncm
```

### 第三步：获取转换后的文件
- 转换后的文件会保存在**原文件同目录下**
- 文件名格式：`艺术家 - 歌曲名.mp3` 或 `.flac`
- 支持的输出格式：MP3、FLAC（根据原始文件决定）

## 转换示例

```bash
# 假设你有这些文件：
# ~/Music/周杰伦 - 稻香.ncm
# ~/Music/邓紫棋 - 光年之外.ncm

# 转换单个文件
python src/ncm_converter.py "~/Music/周杰伦 - 稻香.ncm"

# 转换整个文件夹
python src/ncm_converter.py ~/Music/

# 转换结果：
# ✅ 转换成功: ~/Music/周杰伦 - 稻香.mp3
# ✅ 转换成功: ~/Music/邓紫棋 - 光年之外.mp3
```

## 现在就可以使用了！

转换完成后，你就可以：

1. **上传到我们的情感分析系统**
   ```bash
   # 启动Web应用
   python src/app.py
   
   # 访问 http://localhost:5000
   # 上传转换后的 .mp3 或 .flac 文件
   ```

2. **在任何音乐播放器中播放**
   - VLC Media Player
   - Windows Media Player  
   - iTunes/Apple Music
   - Spotify (本地文件)

## 注意事项

### ⚠️ 重要提醒
- **仅供个人学习和已购买音乐的格式转换**
- **请尊重音乐版权，勿用于商业用途**
- **不要分享或传播转换后的文件**

### 🔧 技术说明
- 转换过程会保留原始音质
- 支持320kbps MP3和无损FLAC格式
- 自动提取歌曲元数据（艺术家、专辑等）

### 🛠️ 故障排除

#### 转换失败？
```bash
# 检查文件是否为有效的NCM格式
file your_song.ncm

# 应该显示类似：
# your_song.ncm: data
```

#### 依赖问题？
```bash
# 重新安装依赖
pip install --upgrade pycryptodome
```

#### 权限问题？
```bash
# 确保有读写权限
chmod 755 src/ncm_converter.py
```

## 快速开始

```bash
# 一键转换脚本
echo "正在查找NCM文件..."
find ~ -name "*.ncm" -type f | head -5

echo "开始转换..."
python src/ncm_converter.py ~/Music/

echo "转换完成！现在可以上传到情感分析系统了 🎉"
```

现在你的网易云音乐文件就可以在我们的AI情感分析系统中使用了！🎵✨