#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCM格式转换工具
将网易云音乐的.ncm文件转换为标准音频格式
"""

import os
import sys
import json
import base64
import struct
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

class NCMConverter:
    """网易云音乐NCM文件转换器"""
    
    def __init__(self):
        # 网易云音乐的固定密钥
        self.core_key = bytes([
            0x68, 0x7A, 0x48, 0x52, 0x41, 0x6D, 0x73, 0x6F,
            0x35, 0x6B, 0x49, 0x6E, 0x62, 0x61, 0x78, 0x57
        ])
        self.meta_key = bytes([
            0x23, 0x31, 0x34, 0x6C, 0x6A, 0x6B, 0x5F, 0x21,
            0x5C, 0x5D, 0x26, 0x30, 0x55, 0x3C, 0x27, 0x28
        ])
    
    def convert_ncm_file(self, ncm_path):
        """转换单个NCM文件"""
        try:
            with open(ncm_path, 'rb') as f:
                # 读取文件头
                header = f.read(8)
                if header != b'CTENFDAM':
                    print(f"❌ {ncm_path} 不是有效的NCM文件")
                    return False
                
                # 跳过2字节
                f.read(2)
                
                # 读取密钥长度和密钥数据
                key_len = struct.unpack('<I', f.read(4))[0]
                key_data = f.read(key_len)
                
                # 解密RC4密钥
                rc4_key = self._decrypt_rc4_key(key_data)
                if not rc4_key:
                    print(f"❌ 无法解密 {ncm_path} 的RC4密钥")
                    return False
                
                # 读取元数据
                meta_len = struct.unpack('<I', f.read(4))[0]
                meta_data = f.read(meta_len)
                music_info = self._decrypt_metadata(meta_data)
                
                # 跳过CRC和Gap
                f.read(9)
                
                # 读取专辑图片
                img_len = struct.unpack('<I', f.read(4))[0]
                img_data = f.read(img_len)
                
                # 读取音频数据
                audio_data = f.read()
                
                # 解密音频数据
                decrypted_audio = self._decrypt_audio(audio_data, rc4_key)
                
                # 确定输出格式和文件名
                output_path = self._get_output_path(ncm_path, music_info)
                
                # 保存解密后的文件
                with open(output_path, 'wb') as out_f:
                    out_f.write(decrypted_audio)
                
                print(f"✅ 转换成功: {output_path}")
                return True
                
        except Exception as e:
            print(f"❌ 转换失败 {ncm_path}: {e}")
            return False
    
    def _decrypt_rc4_key(self, key_data):
        """解密RC4密钥"""
        try:
            # 与0x64异或
            for i in range(len(key_data)):
                key_data = key_data[:i] + bytes([key_data[i] ^ 0x64]) + key_data[i+1:]
            
            # AES解密
            cipher = AES.new(self.core_key, AES.MODE_ECB)
            decrypted = cipher.decrypt(key_data)
            
            # 去除填充
            decrypted = unpad(decrypted, 16)
            
            # 去除前缀"neteasecloudmusic"
            prefix = b"neteasecloudmusic"
            if decrypted.startswith(prefix):
                return decrypted[len(prefix):]
            return decrypted
            
        except Exception as e:
            print(f"RC4密钥解密错误: {e}")
            return None
    
    def _decrypt_metadata(self, meta_data):
        """解密元数据"""
        try:
            # 与0x63异或
            for i in range(len(meta_data)):
                meta_data = meta_data[:i] + bytes([meta_data[i] ^ 0x63]) + meta_data[i+1:]
            
            # 去除前缀
            prefix = b"163 key(Don't modify):"
            if meta_data.startswith(prefix):
                meta_data = meta_data[len(prefix):]
            
            # Base64解码
            decoded = base64.b64decode(meta_data)
            
            # AES解密
            cipher = AES.new(self.meta_key, AES.MODE_ECB)
            decrypted = cipher.decrypt(decoded)
            
            # 去除填充和前缀
            decrypted = unpad(decrypted, 16)
            prefix = b"music:"
            if decrypted.startswith(prefix):
                decrypted = decrypted[len(prefix):]
            
            # 解析JSON
            return json.loads(decrypted.decode('utf-8'))
            
        except Exception as e:
            print(f"元数据解密错误: {e}")
            return {}
    
    def _decrypt_audio(self, audio_data, rc4_key):
        """解密音频数据"""
        # 初始化RC4
        s_box = list(range(256))
        j = 0
        
        for i in range(256):
            j = (j + s_box[i] + rc4_key[i % len(rc4_key)]) % 256
            s_box[i], s_box[j] = s_box[j], s_box[i]
        
        # 解密数据
        decrypted = bytearray(audio_data)
        for idx in range(len(decrypted)):
            i = (idx + 1) % 256
            j = (s_box[i] + i) % 256
            k = (s_box[i] + s_box[j]) % 256
            decrypted[idx] ^= s_box[k]
        
        return bytes(decrypted)
    
    def _get_output_path(self, ncm_path, music_info):
        """生成输出文件路径"""
        ncm_file = Path(ncm_path)
        
        # 尝试从元数据获取信息
        if music_info:
            format_ext = music_info.get('format', 'mp3')
            artist = music_info.get('artist', [['Unknown']])[0][0] if music_info.get('artist') else 'Unknown'
            title = music_info.get('musicName', 'Unknown')
            filename = f"{artist} - {title}.{format_ext}"
        else:
            # fallback到原文件名
            filename = ncm_file.stem + '.mp3'
        
        # 清理文件名中的非法字符
        filename = self._sanitize_filename(filename)
        
        return ncm_file.parent / filename
    
    def _sanitize_filename(self, filename):
        """清理文件名中的非法字符"""
        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            filename = filename.replace(char, '_')
        return filename
    
    def convert_directory(self, directory_path):
        """转换目录中的所有NCM文件"""
        directory = Path(directory_path)
        ncm_files = list(directory.glob('*.ncm'))
        
        if not ncm_files:
            print(f"❌ 在 {directory_path} 中未找到NCM文件")
            return
        
        print(f"🎵 找到 {len(ncm_files)} 个NCM文件")
        
        success_count = 0
        for ncm_file in ncm_files:
            if self.convert_ncm_file(ncm_file):
                success_count += 1
        
        print(f"\n📊 转换完成: {success_count}/{len(ncm_files)} 个文件转换成功")


def main():
    """主函数"""
    print("🎵 网易云音乐NCM格式转换工具")
    print("=" * 50)
    
    if len(sys.argv) != 2:
        print("使用方法: python ncm_converter.py <NCM文件或目录路径>")
        print("示例:")
        print("  python ncm_converter.py song.ncm")
        print("  python ncm_converter.py /path/to/music/folder/")
        return
    
    input_path = sys.argv[1]
    
    # 检查路径是否存在
    if not os.path.exists(input_path):
        print(f"❌ 路径不存在: {input_path}")
        return
    
    # 检查依赖
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
    except ImportError:
        print("❌ 缺少依赖库，请安装: pip install pycryptodome")
        return
    
    converter = NCMConverter()
    
    if os.path.isfile(input_path) and input_path.endswith('.ncm'):
        # 转换单个文件
        converter.convert_ncm_file(input_path)
    elif os.path.isdir(input_path):
        # 转换目录中的所有文件
        converter.convert_directory(input_path)
    else:
        print("❌ 请提供有效的NCM文件或包含NCM文件的目录")


if __name__ == "__main__":
    main()