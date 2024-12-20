﻿# 利用CNN+LSTM做病毒的封包分析

## 第一步
用沙盒分析所有病毒的封包，記錄Src IP,Dst IP,Src port,Dst port, Transport Protocol這五個tuple，並用16進制讀取
```
python code/read16.py
output 16進
```

## 第二步
把16進制轉換成2進制，把2進制的每8bit當作一格灰階色塊，統計有幾byte後開更號算邊長，多餘的截掉，生成圖片
```
input 16進
python code/2toimage
output 2進
```

## 第三步
把所有圖片大小轉成64*64
```
input 2進
python code/resize.py
output image
```

## 第四步
用cnn+lstm模型訓練分類
```
input image
python code/lstmcnn2.py
```

4-1以下是訓練出來的結果
![Accuracy](/results/training_history.png)
