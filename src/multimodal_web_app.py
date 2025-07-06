#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态音乐情感识别Web应用
支持音频+歌词的双重情感分析
"""

import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import base64
import io
import shutil

# 导入中文字体配置
from matplotlib_chinese_config import setup_chinese_matplotlib
setup_chinese_matplotlib()

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from multimodal_predictor import MultiModalMusicEmotionPredictor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 中文字体已在matplotlib_chinese_config中配置

# 初始化多模态预测器
try:
    multimodal_predictor = MultiModalMusicEmotionPredictor()
    logger.info("多模态预测器初始化成功")
except Exception as e:
    logger.error(f"多模态预测器初始化失败: {e}")
    multimodal_predictor = None

def create_simplified_visualization(data):
    """为概率全为0的情况创建简化的可视化图表"""
    try:
        # 创建简化图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 显示预测结果
        ax1.axis('off')
        ax1.text(0.5, 0.7, f"预测情感: {data.get('predicted_emotion', 'N/A')}", 
                ha='center', va='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
        ax1.text(0.5, 0.5, f"置信度: {data.get('confidence', 0):.1%}", 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.text(0.5, 0.3, f"分析方法: {data.get('modality', 'N/A')}", 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('分析结果', fontsize=14, fontweight='bold')
        
        # 2. 显示说明信息
        ax2.axis('off')
        explanation_text = [
            "仅歌词分析结果说明:",
            "",
            "• 基于歌词文本的情感分析",
            "• 使用中文情感词典匹配",
            "• 分析文本的情感倾向",
            "• 可能需要更多情感词汇",
            "  来提高分析准确性"
        ]
        
        for i, text in enumerate(explanation_text):
            ax2.text(0.1, 0.9 - i*0.12, text, fontsize=10, transform=ax2.transAxes,
                    verticalalignment='top')
        ax2.set_title('分析说明', fontsize=14, fontweight='bold')
        
        # 3. 显示歌词关键词（如果有）
        ax3.axis('off')
        if 'lyrics_result' in data and data['lyrics_result'].get('emotion_words_found'):
            emotion_words = [word['word'] for word in data['lyrics_result']['emotion_words_found'][:10]]
            if emotion_words:
                ax3.text(0.1, 0.9, "发现的情感词汇:", fontsize=12, fontweight='bold',
                        transform=ax3.transAxes)
                for i, word in enumerate(emotion_words):
                    row = i // 3
                    col = i % 3
                    ax3.text(0.1 + col*0.3, 0.8 - row*0.1, f"• {word}", fontsize=10,
                            transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, "未发现明显的情感词汇", ha='center', va='center',
                        transform=ax3.transAxes, fontsize=12)
        else:
            ax3.text(0.5, 0.5, "无歌词分析数据", ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
        ax3.set_title('关键词分析', fontsize=14, fontweight='bold')
        
        # 4. 显示建议
        ax4.axis('off')
        suggestions = [
            "分析建议:",
            "",
            "✓ 尝试使用更丰富的情感词汇",
            "✓ 添加更多描述性的语言",
            "✓ 结合音频分析获得更好效果",
            "✓ 使用多模态分析提高准确性"
        ]
        
        for i, text in enumerate(suggestions):
            ax4.text(0.1, 0.9 - i*0.12, text, fontsize=10, transform=ax4.transAxes,
                    verticalalignment='top')
        ax4.set_title('改进建议', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 转换为base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'timestamp': datetime.now().isoformat(),
            'type': 'simplified'
        })
        
    except Exception as e:
        logger.error(f"简化可视化生成失败: {e}")
        return jsonify({'error': f'可视化生成失败: {str(e)}'}), 500

def allowed_file(filename):
    """检查文件是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['mp3', 'wav', 'flac', 'm4a']

@app.route('/')
def index():
    """主页面"""
    return render_template('multimodal_emotion_app.html')

@app.route('/apple')
def apple_index():
    """苹果风格主页面"""
    return render_template('apple_style_multimodal_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理多模态预测请求"""
    logger.info("=== 开始处理多模态预测请求 ===")
    
    if multimodal_predictor is None:
        logger.error("多模态预测器未初始化")
        return jsonify({'error': 'Multimodal predictor not initialized'}), 500
    
    # 获取歌词
    lyrics = request.form.get('lyrics', '').strip()
    
    # 获取音频文件
    audio_file = request.files.get('audio')
    
    # 检查是否至少有一种输入
    if not lyrics and (not audio_file or audio_file.filename == ''):
        return jsonify({'error': '请至少提供音频文件或歌词'}), 400
    
    try:
        audio_path = None
        
        # 处理音频文件
        if audio_file and audio_file.filename and audio_file.filename != '' and allowed_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{filename}"
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            
            audio_file.save(audio_path)
            logger.info(f"音频文件已保存: {audio_path}")
        
        # 进行多模态预测
        result = multimodal_predictor.predict_adaptive(audio_path or '', lyrics)
        
        # 添加解释
        result['explanation'] = multimodal_predictor.get_prediction_explanation(result)
        
        # 清理临时文件
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info("临时音频文件已清理")
        
        logger.info(f"多模态预测成功: {result['predicted_emotion']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"多模态预测失败: {e}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/lyrics_only', methods=['POST'])
def predict_lyrics_only():
    """仅基于歌词的情感预测"""
    logger.info("=== 处理纯歌词预测请求 ===")
    
    if multimodal_predictor is None:
        return jsonify({'error': 'Multimodal predictor not initialized'}), 500
    
    lyrics = request.form.get('lyrics', '').strip()
    
    if not lyrics:
        return jsonify({'error': '请输入歌词'}), 400
    
    try:
        result = multimodal_predictor.predict_lyrics_only(lyrics)
        result['explanation'] = multimodal_predictor.get_prediction_explanation(result)
        
        logger.info(f"歌词预测成功: {result['predicted_emotion']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"歌词预测失败: {e}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/audio_only', methods=['POST'])
def predict_audio_only():
    """仅基于音频的情感预测"""
    logger.info("=== 处理纯音频预测请求 ===")
    
    if multimodal_predictor is None:
        return jsonify({'error': 'Multimodal predictor not initialized'}), 500
    
    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '' or not allowed_file(audio_file.filename):
        return jsonify({'error': '请上传有效的音频文件'}), 400
    
    try:
        if not audio_file.filename:
            return jsonify({'error': '音频文件名无效'}), 400
        
        filename = secure_filename(audio_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        audio_file.save(audio_path)
        logger.info(f"音频文件已保存: {audio_path}")
        
        result = multimodal_predictor.predict_audio_only(audio_path)
        result['explanation'] = multimodal_predictor.get_prediction_explanation(result)
        
        # 清理临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info("临时音频文件已清理")
        
        logger.info(f"音频预测成功: {result['predicted_emotion']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"音频预测失败: {e}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    """生成预测结果的可视化图表"""
    try:
        data = request.get_json()
        
        if not data or 'all_probabilities' not in data:
            return jsonify({'error': '缺少必要的预测数据'}), 400
        
        # 检查概率数据是否有效
        probabilities = list(data['all_probabilities'].values())
        if all(prob == 0 for prob in probabilities):
            # 如果所有概率都是0，创建一个简化的图表
            return create_simplified_visualization(data)
        
        # 中文字体已在全局配置中设置
        
        # 创建图表 - 修复雷达图的创建
        fig = plt.figure(figsize=(16, 12))
        
        # 创建子图布局
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # 柱状图
        ax2 = fig.add_subplot(gs[0, 1])  # 饼图
        ax3 = fig.add_subplot(gs[0, 2])  # 分析详情
        ax4 = fig.add_subplot(gs[1, :], projection='polar')  # 雷达图占据底部全部空间
        
        # 1. 情感概率分布（柱状图）
        emotions = list(data['all_probabilities'].keys())
        probabilities = list(data['all_probabilities'].values())
        
        colors = ['#FFD700', '#90EE90', '#FF6347', '#9370DB', '#FF69B4', '#4B0082', '#CD853F', '#DC143C']
        bars = ax1.bar(emotions, probabilities, color=colors[:len(emotions)])
        ax1.set_title('情感概率分布', fontsize=14, fontweight='bold')
        ax1.set_ylabel('概率')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 情感概率分布（饼图）
        ax2.pie(probabilities, labels=emotions, autopct='%1.1f%%', colors=colors[:len(emotions)])
        ax2.set_title('情感分布饼图', fontsize=14, fontweight='bold')
        
        # 3. 分析方法信息
        ax3.axis('off')
        
        # 创建分析信息文本
        info_text = []
        info_text.append(f"预测情感: {data.get('predicted_emotion', 'N/A')}")
        info_text.append(f"置信度: {data.get('confidence', 0):.1%}")
        info_text.append(f"分析方法: {data.get('modality', 'N/A')}")
        
        if 'prediction_strategy' in data:
            strategy_names = {
                'audio_only_fallback': '仅音频分析',
                'lyrics_only_fallback': '仅歌词分析',
                'multimodal_optimal': '多模态融合分析'
            }
            strategy = strategy_names.get(data['prediction_strategy'], data['prediction_strategy'])
            info_text.append(f"预测策略: {strategy}")
        
        if 'fusion_weights' in data:
            weights = data['fusion_weights']
            info_text.append(f"音频权重: {weights.get('audio', 0):.1%}")
            info_text.append(f"歌词权重: {weights.get('lyrics', 0):.1%}")
        
        # 显示歌词分析结果
        if 'lyrics_result' in data and data['lyrics_result'].get('emotion_words_found'):
            emotion_words = [word['word'] for word in data['lyrics_result']['emotion_words_found'][:5]]
            if emotion_words:
                info_text.append(f"关键情感词: {', '.join(emotion_words)}")
        
        # 在第三个子图中显示信息
        for i, text in enumerate(info_text):
            ax3.text(0.1, 0.9 - i*0.1, text, fontsize=12, transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax3.set_title('分析详情', fontsize=14, fontweight='bold')
        
        # 4. 情感雷达图（修复极坐标问题）
        if len(emotions) >= 3:  # 至少需要3个情感类别
            angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
            angles += angles[:1]  # 闭合圆形
            
            values = probabilities + [probabilities[0]]  # 闭合数据
            
            # 使用极坐标图的正确方法
            try:
                # 使用动态属性访问避免类型检查错误
                theta_offset = getattr(ax4, 'set_theta_offset', None)
                if theta_offset:
                    theta_offset(np.pi / 2)  # 从顶部开始
                
                theta_direction = getattr(ax4, 'set_theta_direction', None)
                if theta_direction:
                    theta_direction(-1)  # 顺时针
                
                ax4.plot(angles, values, 'o-', linewidth=2, color='blue', markersize=8)
                ax4.fill(angles, values, alpha=0.25, color='blue')
                
                # 设置角度标签
                thetagrids = getattr(ax4, 'set_thetagrids', None)
                if thetagrids:
                    thetagrids(np.degrees(angles[:-1]), emotions)
                
                ax4.set_title('情感雷达图', fontsize=14, fontweight='bold', pad=20)
                ax4.set_ylim(0, max(probabilities) * 1.2)
                
                # 添加网格
                ax4.grid(True)
                
                # 为每个点添加数值标签
                for angle, value, emotion in zip(angles[:-1], values[:-1], emotions):
                    ax4.text(angle, value + max(probabilities) * 0.1, f'{value:.1%}', 
                            ha='center', va='center', fontweight='bold', fontsize=10)
            except Exception:
                # 如果极坐标方法不可用，显示错误信息
                ax4.axis('off')
                ax4.text(0.5, 0.5, '极坐标雷达图不可用\n请检查matplotlib版本', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, '情感类别不足\n无法显示雷达图', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        # 检查是否请求下载
        download_requested = data.get('download', False)
        
        if download_requested:
            # 保存图表文件用于下载
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(tempfile.gettempdir(), f'emotion_chart_{timestamp}.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 发送文件下载
            return send_file(chart_path, as_attachment=True, download_name='emotion_analysis.png')
        else:
            # 转换为base64用于页面显示
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return jsonify({
                'success': True,
                'image': image_base64,
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
        return jsonify({'error': f'可视化生成失败: {str(e)}'}), 500

@app.route('/compare', methods=['POST'])
def compare_analyses():
    """比较不同分析方法的结果"""
    try:
        data = request.get_json()
        
        if not data or 'results' not in data:
            return jsonify({'error': '缺少比较数据'}), 400
        
        results = data['results']
        
        # 创建比较图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 预测结果对比
        methods = []
        predictions = []
        confidences = []
        
        for result in results:
            method_name = {
                'audio_only': '仅音频',
                'lyrics_only': '仅歌词',
                'multimodal': '多模态融合'
            }.get(result.get('modality', 'unknown'), '未知')
            
            methods.append(method_name)
            predictions.append(result.get('predicted_emotion', 'N/A'))
            confidences.append(result.get('confidence', 0))
        
        bars = ax1.bar(methods, confidences, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('不同方法的预测置信度对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('置信度')
        ax1.set_ylim(0, 1)
        
        # 在柱状图上添加预测结果标签
        for i, (bar, pred, conf) in enumerate(zip(bars, predictions, confidences)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{pred}\n{conf:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 情感概率热力图
        if len(results) > 1:
            # 构建热力图数据
            all_emotions = set()
            for result in results:
                all_emotions.update(result.get('all_probabilities', {}).keys())
            
            all_emotions = sorted(list(all_emotions))
            heatmap_data = []
            
            for result in results:
                probs = result.get('all_probabilities', {})
                row = [probs.get(emotion, 0) for emotion in all_emotions]
                heatmap_data.append(row)
            
            im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax2.set_xticks(range(len(all_emotions)))
            ax2.set_xticklabels(all_emotions, rotation=45)
            ax2.set_yticks(range(len(methods)))
            ax2.set_yticklabels(methods)
            ax2.set_title('情感概率热力图', fontsize=14, fontweight='bold')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label('概率值')
            
            # 在热力图上添加数值
            for i in range(len(methods)):
                for j in range(len(all_emotions)):
                    text = ax2.text(j, i, f'{heatmap_data[i][j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=10)
        
        # 3. 方法特点比较
        ax3.axis('off')
        
        comparison_text = [
            "分析方法特点比较:",
            "",
            "🎵 仅音频分析:",
            "  • 基于音频特征（节拍、音色、频谱等）",
            "  • 适用于无歌词的音乐",
            "  • 客观性强，不受语言理解影响",
            "",
            "📝 仅歌词分析:",
            "  • 基于文本语义和情感词典",
            "  • 适用于有明确歌词的音乐",
            "  • 能捕捉文字表达的情感",
            "",
            "🎭 多模态融合:",
            "  • 结合音频和歌词信息",
            "  • 提供更全面的情感判断",
            "  • 准确性通常最高"
        ]
        
        for i, text in enumerate(comparison_text):
            ax3.text(0.05, 0.95 - i*0.05, text, fontsize=10, transform=ax3.transAxes,
                    verticalalignment='top')
        
        ax3.set_title('方法特点对比', fontsize=14, fontweight='bold')
        
        # 4. 一致性分析
        ax4.axis('off')
        
        if len(results) > 1:
            # 计算预测一致性
            unique_predictions = set(predictions)
            consistency_score = 1 - (len(unique_predictions) - 1) / len(results)
            
            # 计算置信度差异
            confidence_std = np.std(confidences)
            
            consistency_text = [
                "预测一致性分析:",
                "",
                f"预测结果: {', '.join(unique_predictions)}",
                f"一致性评分: {consistency_score:.1%}",
                f"置信度标准差: {confidence_std:.3f}",
                "",
                "建议:"
            ]
            
            if consistency_score > 0.8:
                consistency_text.append("✅ 各方法预测结果高度一致")
            elif consistency_score > 0.5:
                consistency_text.append("⚠️ 预测结果存在分歧，建议参考多模态结果")
            else:
                consistency_text.append("❌ 预测结果差异较大，建议重新分析")
            
            for i, text in enumerate(consistency_text):
                ax4.text(0.05, 0.95 - i*0.08, text, fontsize=11, transform=ax4.transAxes,
                        verticalalignment='top')
        
        ax4.set_title('一致性分析', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存比较图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = f'temp_comparison_{timestamp}.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 发送图片文件
        return send_file(chart_path, as_attachment=True, download_name='emotion_comparison.png')
        
    except Exception as e:
        logger.error(f"比较分析失败: {e}")
        return jsonify({'error': f'比较分析失败: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """收集用户反馈"""
    try:
        data = request.get_json()
        
        # 保存反馈数据
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'predicted_emotion': data.get('predicted_emotion'),
            'actual_emotion': data.get('actual_emotion'),
            'confidence': data.get('confidence'),
            'modality': data.get('modality'),
            'user_rating': data.get('user_rating'),
            'user_comment': data.get('user_comment', '')
        }
        
        # 写入反馈文件
        feedback_file = 'data/multimodal_feedback.jsonl'
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        
        logger.info(f"用户反馈已收集: {feedback_data}")
        return jsonify({'status': 'success', 'message': '反馈已收集，谢谢！'})
        
    except Exception as e:
        logger.error(f"反馈收集失败: {e}")
        return jsonify({'error': f'反馈收集失败: {str(e)}'}), 500

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'multimodal_predictor': multimodal_predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🎵 启动多模态音乐情感识别Web应用...")
    print("📊 支持功能:")
    print("  - 音频情感分析")
    print("  - 歌词情感分析")
    print("  - 多模态融合分析")
    print("  - 可视化图表")
    print("  - 方法对比")
    print("🌐 访问地址: http://localhost:5001")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 