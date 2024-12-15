import os
from PIL import Image

def resize_image(input_image_path, output_image_path, size=(64, 64)):
    """將圖片縮放到指定大小並保存"""
    with Image.open(input_image_path) as img:
        # 使用 LANCZOS 作為高品質縮放方式
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        # 保存縮放後的圖片
        resized_img.save(output_image_path)
        print(f"圖片已縮放並保存: {output_image_path}")

def batch_resize_images(input_dir, output_dir, size=(64, 64)):
    """批量縮放目錄中的所有圖片"""
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍歷輸入目錄中的所有文件
    for file_name in os.listdir(input_dir):
        # 檢查是否是圖片文件
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_image_path = os.path.join(input_dir, file_name)
            output_image_path = os.path.join(output_dir, file_name)
            
            print(f"正在處理圖片: {input_image_path}")
            resize_image(input_image_path, output_image_path, size)

if __name__ == "__main__":
    # 輸入圖片目錄
    input_dir = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\2進\vtflooder"  # 替換為您的輸入目錄路徑
    # 輸出圖片目錄
    output_dir = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\2進\train\vtflooder"  # 替換為您的輸出目錄路徑
    
    print("批量縮放圖片開始...")
    batch_resize_images(input_dir, output_dir, size=(64, 64))
    print("所有圖片處理完成！")
