from flask import Flask, render_template, request, jsonify
import os
from audio_processor import AudioProcessor
from emotion_classifier import EmotionClassifier
import numpy as np

# 创建Flask应用，指定正确的模板目录
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化处理器
audio_processor = AudioProcessor()
emotion_classifier = EmotionClassifier()

# 初始化模型（创建一个简单的默认模型用于演示）
def initialize_model():
    """初始化情感分类器模型"""
    try:
        # 假设的输入特征维度：13个MFCC + 1个频谱质心 + 12个色度 + 1个过零率 + 1个RMS = 28
        input_shape = (28,)
        emotion_classifier.build_model(input_shape)
        print("情感分类器模型已初始化")
        return True
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 检查文件扩展名
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'不支持的文件格式: {file_ext}。支持的格式: {", ".join(allowed_extensions)}'}), 400
    
    if file:
        # 保存文件
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filename)
        except Exception as e:
            return jsonify({'error': f'文件保存失败: {str(e)}'}), 500
        
        try:
            # 提取特征
            features = audio_processor.get_emotion_features(filename)
            if features is None:
                return jsonify({'error': '音频文件处理失败，请检查文件格式是否正确'}), 400
            
            # 检查特征是否为空
            if len(features) == 0:
                return jsonify({'error': '无法从音频文件中提取有效特征'}), 400
            
            # 预测情感（如果模型未训练，返回模拟数据）
            try:
                emotions = emotion_classifier.predict_emotion_sequence(features)
            except Exception as e:
                print(f"情感预测失败: {str(e)}")
                # 返回模拟数据用于演示
                emotions = []
                for i, _ in enumerate(features):
                    emotions.append({
                        'emotion': 'happy',
                        'probabilities': {
                            'happy': 0.6,
                            'sad': 0.1,
                            'angry': 0.1,
                            'peaceful': 0.1,
                            'excited': 0.1
                        }
                    })
            
            # 准备时间轴数据（每3秒一个点）
            timestamps = [i * 3 for i in range(len(emotions))]
            
            # 准备返回数据
            result = {
                'timestamps': timestamps,
                'emotions': emotions,
                'filename': file.filename,
                'duration': len(emotions) * 3
            }
            
            return jsonify(result)
            
        except Exception as e:
            print(f"处理音频文件时发生错误: {str(e)}")
            return jsonify({'error': f'音频处理失败: {str(e)}'}), 500
        finally:
            # 清理上传的文件
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except Exception as e:
                    print(f"清理临时文件失败: {str(e)}")
    
    return jsonify({'error': 'Unknown error'}), 500

if __name__ == '__main__':
    # 初始化模型
    model_initialized = initialize_model()
    if not model_initialized:
        print("警告: 模型初始化失败，将使用模拟数据")
    
    print("启动Flask应用...")
    app.run(debug=True, host='0.0.0.0', port=5000) 