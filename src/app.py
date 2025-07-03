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
    
    if file:
        # 保存文件
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        try:
            # 提取特征
            features = audio_processor.get_emotion_features(filename)
            if features is None:
                return jsonify({'error': 'Failed to process audio file'}), 400
            
            # 预测情感
            emotions = emotion_classifier.predict_emotion_sequence(features)
            
            # 准备时间轴数据（每3秒一个点）
            timestamps = [i * 3 for i in range(len(emotions))]
            
            # 准备返回数据
            result = {
                'timestamps': timestamps,
                'emotions': emotions,
                'filename': file.filename
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # 清理上传的文件
            if os.path.exists(filename):
                os.remove(filename)
    
    return jsonify({'error': 'Unknown error'}), 500

if __name__ == '__main__':
    # 加载模型
    input_shape = (25,)  # 13 MFCC + 1 spectral centroid + 12 chroma + 1 zero crossing rate + 1 rmse
    emotion_classifier.build_model(input_shape)
    # TODO: 在这里加载预训练模型权重
    
    app.run(debug=True) 