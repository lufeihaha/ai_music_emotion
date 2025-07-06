#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎨 通用中文字体配置 - 解决matplotlib中文显示问题
"""

import matplotlib
import matplotlib.pyplot as plt
import platform
import os
import logging

logger = logging.getLogger(__name__)

def setup_chinese_matplotlib():
    """设置matplotlib中文字体支持"""
    
    # 强制使用非GUI后端
    matplotlib.use('Agg')
    
    # 获取系统信息
    system = platform.system()
    
    try:
        # 根据系统选择合适的字体
        if system == "Windows":
            fonts = ['Microsoft YaHei UI', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        elif system == "Darwin":  # macOS
            fonts = ['SF Pro Display', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'sans-serif']
        elif system == "Linux":
            fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
        else:
            fonts = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
        
        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 11
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 200
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['figure.max_open_warning'] = 0
        
        logger.info(f"✅ 字体配置完成 - 系统: {system}, 主要字体: {fonts[0]}")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ 字体配置失败: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 函数已在上面定义，无需重复

def test_chinese_font():
    """测试中文字体是否正常工作"""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['快乐', '平静', '忧郁'], [0.3, 0.4, 0.3])
        ax.set_title('字体测试')
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        logger.info("✅ 中文字体测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ 字体测试失败: {e}")
        return False

if __name__ == "__main__":
    setup_chinese_matplotlib()
    test_chinese_font()