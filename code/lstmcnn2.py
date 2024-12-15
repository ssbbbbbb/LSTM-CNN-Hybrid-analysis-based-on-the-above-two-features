import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib import rcParams


def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()

    # 第一層 CNN
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # 第二層 CNN
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # 展平
    model.add(Flatten())

    # Reshape 為 LSTM 輸入格式
    flattened_dim = model.output_shape[1]
    model.add(Reshape((1, flattened_dim)))

    # LSTM 部分
    model.add(LSTM(64, return_sequences=False))

    # 全連接層
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def plot_history(history, save_path):
    # 設置支持中文的字體
    rcParams['font.family'] = 'Microsoft JhengHei'  # 替換為你系統中存在的中文支持字體
    rcParams['axes.unicode_minus'] = False  # 允許負號顯示

    # 繪製損失和準確率曲線
    plt.figure(figsize=(12, 4))

    # 損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='訓練損失')
    plt.plot(history.history['val_loss'], label='驗證損失')
    plt.title('損失曲線')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='訓練準確率')
    plt.plot(history.history['val_accuracy'], label='驗證準確率')
    plt.title('準確率曲線')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()


def train_model(data_dir, output_dir, image_size=(64, 64), batch_size=16, epochs=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 資料生成器（加入資料增強）
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3,  # 分出 30% 作為驗證集
        rotation_range=30,      # 隨機旋轉角度
        width_shift_range=0.2,  # 隨機水平平移
        height_shift_range=0.2, # 隨機垂直平移
        shear_range=0.2,        # 隨機剪切
        zoom_range=0.2,         # 隨機縮放
        horizontal_flip=True,   # 隨機水平翻轉
        fill_mode='nearest'     # 填充方式
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True  # 確保訓練數據被隨機打亂
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    input_shape = image_size + (3,)

    model = build_cnn_lstm_model(input_shape, num_classes)

    # 設定初始學習率為 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 回調函數：保存最佳模型和記錄訓練過程
    checkpoint_path = os.path.join(output_dir, 'best_model.keras')  # 保存整個模型
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    # 學習率調度器：當驗證損失不再改善時，降低學習率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,            # 學習率下降的因子
                                  patience=5,            # 當驗證損失在 `patience` 個 epoch 內沒有改善時，觸發
                                  verbose=1,
                                  min_lr=1e-9)           # 最小學習率

    # CSV 日誌記錄
    csv_logger = CSVLogger(os.path.join(output_dir, 'training_log.csv'), append=True)

    callbacks = [checkpoint, reduce_lr, csv_logger]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # 繪製並保存訓練曲線
    plot_history(history, output_dir)

    # 保存最終模型權重
    final_weights_path = os.path.join(output_dir, 'final_model.weights.h5')
    model.save_weights(final_weights_path)
    print(f"模型權重已保存至 {output_dir}")


if __name__ == "__main__":
    # 使用者指定的資料路徑和輸出路徑
    data_directory = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\2進\train1"  # 替換為你的圖片資料夾路徑
    output_directory = r"C:\Users\蕭宗賓\Desktop\AI local\work\動態2\results"   # 替換為你想儲存結果的資料夾路徑

    train_model(data_directory, output_directory, image_size=(64, 64), batch_size=1, epochs=100)
