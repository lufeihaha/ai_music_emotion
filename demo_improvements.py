#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音乐情感识别系统改进演示脚本
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🎵 音乐情感识别系统改进演示")
    print("=" * 50)
    
    print("\n📋 可用的改进选项:")
    print("1. 🗂️  数据增强 - 扩展训练数据")
    print("3. 🏷️  情感标签细化 - 支持更多情感类别")
    print("4. 📊 查看改进指南")
    print("5. 🧪 测试当前模型")
    print("0. 🚪 退出")
    
    while True:
        choice = input("\n请选择改进选项 (0-5): ").strip()
        
        if choice == "1":
            print("\n🔄 运行数据增强...")
            print("💡 这将为现有音频数据生成变体以提高模型泛化能力")
            confirm = input("是否继续? (y/n): ").lower()
            if confirm == 'y':
                try:
                    os.system("python src/data_enhancement.py")
                except Exception as e:
                    print(f"❌ 数据增强失败: {e}")
                    print("💡 请确保已安装所需依赖: pip install soundfile tqdm")
            
        elif choice == "3":
            print("\n🏷️ 情感标签细化...")
            print("当前支持的情感:")
            emotions = {
                'calm': '🍃 平静',
                'happy': '😊 快乐', 
                'energetic': '⚡ 充满活力',
                'melancholic': '☁️ 忧郁'
            }
            
            for key, name in emotions.items():
                print(f"  - {name}")
            
            print("\n建议新增的细化情感:")
            new_emotions = {
                'romantic': '💕 浪漫',
                'nostalgic': '⏰ 怀念',
                'dramatic': '🎭 戏剧性',
                'mysterious': '🌙 神秘'
            }
            
            for key, name in new_emotions.items():
                print(f"  - {name}")
                
            confirm = input("\n是否运行高级情感分类器? (y/n): ").lower()
            if confirm == 'y':
                try:
                    os.system("python src/advanced_emotion_classifier.py")
                except Exception as e:
                    print(f"❌ 高级分类器运行失败: {e}")
            
        elif choice == "4":
            print("\n📖 改进指南已生成: IMPROVEMENT_GUIDE.md")
            print("请查看该文件了解详细的改进方案")
            
        elif choice == "5":
            print("\n🧪 测试当前模型...")
            print("推荐测试歌曲:")
            test_songs = [
                "林俊杰 - 可惜没如果 (当前识别: 平静, 期望: 怀念)",
                "Taylor Swift - Last Christmas (当前识别: 充满活力)",
                "轻音乐或古典音乐 (期望: 平静)",
                "摇滚或电子音乐 (期望: 充满活力)"
            ]
            
            for i, song in enumerate(test_songs, 1):
                print(f"  {i}. {song}")
            
            print("\n💡 访问 http://localhost:5000 测试你的音乐文件")
            confirm = input("是否启动Web应用? (y/n): ").lower()
            if confirm == 'y':
                try:
                    os.system("python run_app.py")
                except Exception as e:
                    print(f"❌ 启动Web应用失败: {e}")
            
        elif choice == "0":
            print("\n👋 再见！")
            break
            
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main() 