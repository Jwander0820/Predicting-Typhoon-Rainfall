# Predicting-Typhoon-Rainfall

## 專案用途

以多源影像神經網路預測颱風期間降雨量的研究專案，本研究使用多種氣象影像作為輸入，預測未來時間點的雨量分布標籤。

主要影像來源：

- `RA`：雨量分布圖
![image_RA](https://github.com/Jwander0820/Predicting-Typhoon-Rainfall/blob/master/img/201609271130_RA.png)
- `RD`：雷達回波圖
![image_RD](https://github.com/Jwander0820/Predicting-Typhoon-Rainfall/blob/master/img/201609271130_RD.png)
- `IR`：紅外線雲圖
![image_IR](https://github.com/Jwander0820/Predicting-Typhoon-Rainfall/blob/master/img/201609271130_IR.png)
- `GI`：經緯度輔助圖，包含 `lon.png` 與 `lat.png`
- 預測出來的雨量分布圖 (RA+RD+IR)
![image_pred](https://github.com/Jwander0820/Predicting-Typhoon-Rainfall/blob/master/img/201609271030_t%2B1_predict.png)

目前主要支援：

- 模型訓練
- 驗證集評估
- 離線推論
- 實驗性即時氣象圖 crawler 推論

## 環境版本

本專案以 TensorFlow `2.9.1` 為固定基準，請不要任意升級 TensorFlow 主版本，避免影響舊 `.h5` 權重載入與研究結果重現。

建議環境：

```text
Python      3.9.x
TensorFlow  2.9.1
```

安裝依賴：

```bash
pip install -r requirement.txt
```

## 快速開始

### 1. 執行單元測試

先確認專案基本設定與資料契約正常：

```bash
python -m unittest discover -s tests -v
```

目前測試不會真的訓練模型，也不依賴外網。

### 2. 執行少量推論

建議第一次確認模型與環境時，先跑驗證集前 25 筆：

```bash
python -m typhoon_rainfall predict --config configs/default.json --limit 25
```

PyCharm 使用者可以直接執行：

```text
run_predict_sample.py
```

執行時 console 會印出本次使用的模型、checkpoint、輸入模態、資料 split 與輸出位置。

### 3. 執行完整驗證

```bash
python -m typhoon_rainfall evaluate --config configs/default.json
```

PyCharm 使用者可以直接執行：

```text
run_evaluate.py
```

### 4. 執行訓練

```bash
python -m typhoon_rainfall train --config configs/default.json
```

PyCharm 使用者可以直接執行：

```text
run_train.py
```

訓練會花較長時間，第一次測試環境時建議先跑 `run_predict_sample.py`。


建議執行順序：

| 檔案 | 用途          | 建議時機                        |
| --- |-------------|-----------------------------|
| `run_predict_sample.py` | 推論驗證集前 25 筆 | 第一次測試最推薦                    |
| `run_evaluate.py` | 跑完整驗證集      | 模型與資料確認後                    |
| `run_train.py` | 開始訓練        | 環境完整且準備訓練時                  |
| `predict_1hour_rainfall.py` | 即時抓圖推論      | 實驗性功能(氣象局資料路徑已改，現已無法抓到最新資料) |


## CLI 使用方式

正式入口為：

```bash
python -m typhoon_rainfall <command> --config configs/default.json
```

可用 command：

```bash
python -m typhoon_rainfall train --config configs/default.json
python -m typhoon_rainfall evaluate --config configs/default.json
python -m typhoon_rainfall predict --config configs/default.json --limit 5
```

`--config` 也可以放在 command 前面：

```bash
python -m typhoon_rainfall --config configs/default.json predict --limit 5
```

若要指定模型，例如改用 `Unet3`：

```bash
python -m typhoon_rainfall predict --config configs/default.json --model-name Unet3 --limit 5
```

若要指定 checkpoint：

```bash
python -m typhoon_rainfall predict \
  --config configs/default.json \
  --checkpoint checkpoint/Unet3_val_min_RMSE.h5 \
  --limit 5
```

## 核心模組

### `typhoon_rainfall/config.py`

集中管理研究設定：

- `DataConfig`：資料路徑、影像大小、輸入模態、預測時間位移
- `ModelConfig`：模型名稱、類別數、checkpoint 設定
- `TrainConfig`：epoch、batch size、learning rate、輸出位置
- `PredictConfig`：推論輸出位置、split、limit
- `ProjectConfig`：整合完整專案設定

### `typhoon_rainfall/data/`

負責資料契約與資料載入：

- `contracts.py`：`SamplePair`、`DatasetPaths`、輸入 shape 計算
- `dataset.py`：讀取 RD / IR / RA / GI，融合 channel，產生 one-hot target

### `typhoon_rainfall/models/`

依照模型名稱建立模型，目前支援：

- `Unet3`
- `Unet4`
- `Unet5`
- `Unet6`
- `FCN8`
- `pspnet`

### `typhoon_rainfall/training/`

負責訓練流程：

- 建立模型
- compile loss 與 metrics
- 建立訓練與驗證 generator
- 儲存 checkpoint
- 輸出 history CSV 與訓練曲線

### `typhoon_rainfall/inference/`

負責推論流程：

- 選擇 checkpoint
- 載入模型權重
- 執行模型推論
- 將 softmax 輸出轉為 label map
- 儲存 CSV 與 PNG

### `typhoon_rainfall/evaluation/`

負責驗證集推論與摘要輸出。

### `typhoon_rainfall/experimental/`

目前包含即時氣象圖 crawler。此功能依賴外部網站格式，屬於實驗功能，不是核心穩定流程。

## 資料格式

預設資料根目錄：

```text
database/
```

請先將 `database.zip` 解壓縮到專案根目錄，解壓縮後應得到 `database/` 資料夾。`database/` 中包含訓練與驗證用資料，主要有雨量分布圖 `RA`、雷達回波圖 `RD`、紅外線雲圖 `IR`、經緯度代表 `GI` 四種資料。

圖像尺寸皆固定為 `128 x 128 pixel`。這些資料不是原始影像，已經過裁切與縮放處理。原始資料集說明中，`train/` 內包含三場颱風資料，`val/` 內包含一場颱風資料；實際訓練與驗證會依 `train_list/` 中的清單選取 input 與 target。

必要結構：

```text
database/
├── train_data_RD/
│   ├── train/
│   └── val/
├── train_data_IR/
│   ├── train/
│   └── val/
├── train_data_RA/
│   └── EWB01/
│       ├── train/
│       └── val/
├── train_data_GI/
│   ├── lon.png
│   └── lat.png
├── train_list/
│   └── t2t+1/
│       ├── train.txt
│       └── val.txt
├── val_label_csv_true_value/
└── val_label_RA_colorimage/
```

資料內容說明：

| 資料 | 說明 |
| --- | --- |
| `RA` | 雨量分布圖，單通道 8-bit 圖像，內部資料為將真值轉換後的標籤值，例如降雨量 `13.67mm` 會轉換為標籤值 `14` |
| `RD` | 雷達回波圖，三通道圖像，擷取自中央氣象局雷達回波圖 |
| `IR` | 紅外線雲圖，單通道圖像，擷取自中央氣象局紅外線雲圖 |
| `GI` | 經緯度代表，單通道圖像，`lat.png` 表示緯度由低到高 `0-254`，`lon.png` 表示經度由左到右 `0-254` |

其他資料夾用途：

| 資料夾 | 用途 |
| --- | --- |
| `train_list/` | 存放訓練模型時會調用的清單，程式會根據清單選取 input 與 target |
| `val_label_csv_true_value/` | 存放驗證資料的真值，用於計算 RMSE 等指標 |
| `val_label_RA_colorimage/` | 存放驗證資料的彩色雨量分布圖，用於視覺化比較與預測結果差異 |

`train.txt` 與 `val.txt` 每一行格式：

```text
source.png;target.png
```

範例：

```text
201308271130.png;201308271230.png
```

代表使用 `201308271130` 的影像作為輸入，預測 `201308271230` 的雨量標籤。

## 輸入通道規則

目前通道融合規則集中於：

```text
typhoon_rainfall/data/contracts.py
typhoon_rainfall/data/dataset.py
```

| 模態 | 通道數 | 說明 |
| --- | --- | --- |
| RD | 3 | RGB 雷達回波圖 |
| IR | 1 或 3 | 與其他模態合併時為 1；單獨使用時展成 3 |
| RA | 3 | 作為輸入時正規化後複製成 3 channel |
| GI | 2 | `lon` + `lat` |

預設輸入：

```text
RD + IR = 4 channels
```

## Checkpoint 規則

預設 checkpoint 目錄：

```text
checkpoint/
```

系統會優先尋找：

```text
checkpoint/{model_name}_val_min_RMSE.h5
```

例如：

```text
checkpoint/Unet5_val_min_RMSE.h5
checkpoint/Unet3_val_min_RMSE.h5
```

若標準名稱不存在，系統會 fallback 到 `checkpoint/` 中最新的 `.h5` 檔案。

## 輸出位置

推論輸出：

```text
output/predict_array/
output/predict/
```

驗證摘要：

```text
output/evaluation_*.csv
```

訓練輸出：

```text
checkpoint/
logs/
output/model_history_*.csv
output/model_history_*.png
```

## 實驗性即時推論

可執行：

```bash
python predict_1hour_rainfall.py
```

此流程會嘗試：

1. 依目前時間推算對應氣象圖時間
2. 下載 IR 與 RD 圖
3. 裁切並縮放為 `128 x 128`
4. 載入模型推論
5. 輸出 CSV 與 PNG

此功能依賴外部氣象網站，氣象局資料路徑已改，現已無法抓到最新資料。正式研究流程請以離線資料訓練與驗證為主。
