import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

def load_model_and_weights(model_path):
    """
    載入保存的模型。
    
    參數:
        model_path (str): 模型檔案的路徑（.keras 格式）。
    
    回傳:
        model (tf.keras.Model): 載入的模型。
    """
    model = tf.keras.models.load_model(model_path)
    print(f"模型已成功載入自 {model_path}")
    return model

def evaluate_model(model, test_generator):
    """
    評估模型在測試集上的表現。
    
    參數:
        model (tf.keras.Model): 已載入的模型。
        test_generator (tf.keras.preprocessing.image.DirectoryIterator): 測試資料生成器。
    
    回傳:
        results (list): 評估結果 [loss, accuracy]。
    """
    results = model.evaluate(test_generator)
    print(f"測試損失: {results[0]:.4f}")
    print(f"測試準確率: {results[1]:.4f}")
    return results

def predict_and_visualize(model, test_generator, class_indices, save_path, num_images=5):
    """
    在測試集上進行預測並可視化結果。
    
    參數:
        model (tf.keras.Model): 已載入的模型。
        test_generator (tf.keras.preprocessing.image.DirectoryIterator): 測試資料生成器。
        class_indices (dict): 類別索引對應字典。
        save_path (str): 保存預測結果圖片的路徑。
        num_images (int): 要顯示的圖片數量。
    """
    # 反轉 class_indices 以獲得索引到類別名稱的映射
    class_labels = {v: k for k, v in class_indices.items()}
    
    # 取得一些測試圖片和真實標籤
    images, labels = next(test_generator)
    
    # 進行預測
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    # 顯示圖片和預測結果
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        true_label = class_labels[true_classes[i]]
        predicted_label = class_labels[predicted_classes[i]]
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        plt.title(f"真實: {true_label}\n預測: {predicted_label}", color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prediction_samples.png'))
    plt.show()
    print(f"預測結果已保存至 {save_path}")

def test_model(
    model_path,
    test_dir,
    output_dir,
    image_size=(128, 128),
    batch_size=32,
    num_images_to_display=5
):
    """
    測試模型，包括載入模型、評估和可視化預測結果。
    
    參數:
        model_path (str): 保存的模型檔案路徑（.keras 格式）。
        test_dir (str): 測試資料集的資料夾路徑。
        output_dir (str): 保存測試結果的資料夾路徑。
        image_size (tuple): 圖片尺寸。
        batch_size (int): 每批次的圖片數量。
        num_images_to_display (int): 要顯示的預測圖片數量。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 載入模型
    model = load_model_and_weights(model_path)
    
    # 準備測試資料生成器
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # 不打亂順序，方便後續預測對應
    )
    
    # 評估模型
    evaluate_model(model, test_generator)
    
    # 進行預測並可視化
    predict_and_visualize(
        model,
        test_generator,
        test_generator.class_indices,
        output_dir,
        num_images=num_images_to_display
    )

if __name__ == "__main__":
    # 使用者指定的模型路徑、測試資料路徑和輸出路徑
    model_file_path = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\results\best_model.keras"  # 替換為你的最佳模型路徑
    test_data_directory = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\2進\test"          # 替換為你的測試資料夾路徑
    output_directory = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\results"           # 替換為你想儲存測試結果的資料夾路徑
    
    test_model(
        model_path=model_file_path,
        test_dir=test_data_directory,
        output_dir=output_directory,
        image_size=(128, 128),
        batch_size=32,
        num_images_to_display=5
    )
