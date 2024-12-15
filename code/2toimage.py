import os
import re
import numpy as np
from math import floor, sqrt
from PIL import Image

def parse_hex_file(file_path):
    """解析檔案中的十六進制數據"""
    with open(file_path, 'r') as f:
        content = f.read()
    # 使用正則表達式提取所有的十六進制數據
    hex_data = re.findall(r'\b[0-9A-Fa-f]{2}\b', content)
    return hex_data

def hex_to_binary(hex_data):
    """將十六進制數據轉換為二進制數據"""
    binary_data = [bin(int(byte, 16))[2:].zfill(8) for byte in hex_data]
    return binary_data

def save_grayscale_image(binary_data, output_image_path):
    """將二進制數據轉換為灰階圖像並保存"""
    # 將二進制數據分組為每字節一個灰階值（0-255）
    grayscale_values = [int(b, 2) for b in binary_data]
    
    # 計算需要的正方形尺寸
    total_blocks = len(grayscale_values)
    dimension = floor(sqrt(total_blocks))  # 開平方向下取整

    # 截取數據以構成正方形
    trimmed_values = grayscale_values[:dimension * dimension]
    image_array = np.array(trimmed_values, dtype=np.uint8).reshape(dimension, dimension)
    
    # 生成圖片並保存
    image = Image.fromarray(image_array, mode='L')
    image.save(output_image_path)
    return total_blocks, dimension

def batch_process(input_dir, output_dir):
    """批量處理目錄中的所有檔案"""
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍歷輸入目錄中的所有檔案
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):  # 只處理 .txt 文件
            input_file = os.path.join(input_dir, file_name)
            output_image = os.path.join(output_dir, file_name.replace(".txt", ".png"))
            
            print(f"正在處理檔案: {input_file}")
            # 解析十六進制數據
            hex_data = parse_hex_file(input_file)
            # 轉換為二進制數據
            binary_data = hex_to_binary(hex_data)
            # 保存灰階圖片
            total_blocks, dimension = save_grayscale_image(binary_data, output_image)
            
            print(f"完成檔案: {file_name}")
            print(f"總格數: {total_blocks}, 開平方結果（圖片邊長）: {dimension}")
            print(f"圖片已儲存至: {output_image}")
            print("-" * 50)

if __name__ == "__main__":
    # 批量處理的輸入和輸出目錄
    input_dir = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\16進\vtflooder"  # 替換為您的輸入目錄路徑
    output_dir = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\2進\vtflooder"  # 替換為您的輸出目錄路徑
    
    print("批量處理開始...")
    batch_process(input_dir, output_dir)
    print("所有檔案處理完成！")
