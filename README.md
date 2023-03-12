# Predicting-Typhoon-Rainfall
預測颱風時期降雨量
透過結合雨量分布圖(RA)、雷達回波圖(RD)、紅外線雲圖(IR)等影像資料預測颱風侵襲期間的降雨量

主要有三種模型分別為U-net、FCN、PSPNet，其中以U-net模型表現較佳

在圖像融合的部分是採用多通道的概念將圖片做疊加

## 主要環境
tensorflow ~= 2.8.0


## 前置說明
請先將database.zip解壓縮到同一資料中，database資料夾中包含訓練與驗證用的資料
1. 主要包含雨量分布圖(RA)、雷達回波圖(RD)、紅外線雲圖(IR)、經緯度代表(GI) 這四種資料

圖像尺寸大小皆固定為128x128 pixel #註:這些資料非原始資料，已經經過裁切與縮放

其中 train資料夾內包含三場颱風資料(176筆)，val資料夾內包含一場颱風資料(56筆)

    a. 雨量分布圖(RA)；單通道圖像(8bit位元深度)，內部資料為將真值轉換為標籤值 Ex. 降雨量 13.67mm 轉換標籤值為 14
    
    b. 雷達回波圖(RD)；三通道圖像，擷取自中央氣象局的雷達回波圖
    
    c. 紅外線雲圖(IR)；單通道圖像，擷取自中央氣象局的紅外線雲圖
    
    d. 經緯度(IR)；單通道圖像，lat.png => 緯度從低到高0-254；lon.png => 經度從左到右0-254
    
2. train_list資料夾則存放訓練模型時會調用的清單，會根據該清單選取input & target

3. val_label_csv_true_value資料夾則存放驗證資料的真值，用於計算RMSE等

4. val_label_RA_colorimage資料夾則存放驗證資料的彩色雨量分布圖，用於視覺化比較與預測結果的差異

## 訓練模型
訓練模型的流程可以參考 train.ipynb

單純測試範例則可以參考 example.ipynb

Forcast_1hour_rainfall.py 為爬蟲當前時刻二十分鐘前的雷達回波圖、紅外線雲圖 <br>
並將其裁切縮放至128x128大小的圖像，丟到預訓練的模型中，嘗試預測雨量分布情形 <br>
但因為訓練的資料(颱風資料)和爬蟲下來的一般雷達回波圖和紅外線雲圖有差異，(淺藍底vs白底、不同的框線)
所以導致預測的結果為空白的情況，也可能因為預訓練的模型較淺層較小型所以預測表現自然也不佳，
若要改善一般預測的情況可能需要重新訓練模型


### 示意圖
以201609271130之時刻圖像作說明
1. 雨量分布圖(RA)

![image](https://github.com/Jwander0820/Predicting-Typhoon-Rainfall/blob/main/img/201609271130_RA.png)

2. 雷達回波圖(RD)

![image](https://github.com/Jwander0820/Predicting-Typhoon-Rainfall/blob/main/img/201609271130_RD.png)

3. 紅外線雲圖(IR)

![image](https://github.com/Jwander0820/Predicting-Typhoon-Rainfall/blob/main/img/201609271130_IR.png)

4. 預測出來的雨量分布圖 (RA+RD+IR)

![image](https://github.com/Jwander0820/Predicting-Typhoon-Rainfall/blob/main/img/201609271030_t%2B1_predict.png)

