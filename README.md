# Predicting-Typhoon-Rainfall

以 RD 雷達回波、IR 紅外線雲圖、RA 雨量分布與 GI 經緯度圖，展示多模態神經網路的下一小時降雨預測。專案隨附已訓練的輕量 Unet3 權重，一般使用不需要重新訓練。

相關期刊論文：[Spatiotemporal deep learning fusion of radar, infrared, and geospatial data for typhoon rainfall estimation in Taiwan](https://link.springer.com/article/10.1007/s12145-025-01985-9)。

| 輸入 | 內容 | 通道數 |
| --- | --- | ---: |
| RD | RGB雷達回波 | 3 |
| IR | 灰階紅外線雲圖 | 1 |
| RA | 逐時測站雨量經IDW內插，灰階複製三層 | 3 |
| GI | 原始LON／LAT灰階圖 | 2 |
| 合計 | 128×128格網 | 9 |

隨附模型為論文概念的簡易復現示範。歷史測試RMSE與持續性預報接近，高雨量有低估；流程可執行不代表已完成現行天氣的預報能力驗收。[模型與評估摘要](docs/research/multimodal-unet3-2026-11-06.md)

## 安裝與執行

所有流程共用 **Python 3.9.x、TensorFlow 2.9.1** 及同一份 `requirement.txt`。在專案根目錄、同一個Python環境安裝並執行：

```powershell
python -m pip install -r requirement.txt
python run_realtime.py
```

若已使用專案 `.venv`，可直接指定它：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
.\.venv\Scripts\python.exe run_realtime.py
```

CPU可以推論，GPU是可選加速。請保留 `checkpoint/Unet3_val_min_RMSE.h5`、同名 `.metadata.json` 及 `database.zip`。即時流程只讀取ZIP內兩張GI圖，不需要先解壓或準備完整歷史資料。

`run_realtime.py` 不帶參數時會取得測站雨量、產生RA、配對RD／IR、加入GI、載入權重並產生預測，不會自動訓練。預設設定為 `configs/realtime.json`；`configs/default.json` 也使用相同九通道、101類Unet3權重。

## 預報時間與結果

例如取得17:00–18:00的完整觀測RA，配對17:30的RD／IR，輸出 **18:00–19:00累積雨量**。資料發布有延遲，執行完成時預報區間可能已開始；它不是從執行時刻起算60分鐘。程式會拒絕未完成觀測、過期資料及已結束的預報區間。

每次結果放在 `output/realtime/<執行時間>_predict/`：

- `forecast.png`：前一小時觀測與下一小時預測對照。
- `prediction_mm.csv`：逐格預測毫米，海面為NaN；類別100表示100mm以上。
- `manifest.json`：來源、觀測／預報時段、模型指紋與執行狀態。
- `inputs/`：真正送入模型的RD、IR、RA、LON、LAT五張圖。
- `alignment.png`：輸入對齊圖，紫色輪廓只供目視參考。
- `observed_ra.png`：觀測 RA 彩色雨量圖，含毫米色階；只供閱讀，模型仍使用 `inputs/RA.png` 原始標籤。

預設為簡潔輸出。需要原圖、測站數值及所有中間檔時，執行 `python run_realtime.py --detailed-output`。失敗執行保留診斷資料。

整個 `output/` 已被Git忽略。各次結果會保留，不會自動清除。

## 其他操作

```powershell
python run_realtime.py --help
python run_realtime.py fetch
python run_realtime.py rainfall --hours 6
python run_realtime.py ra --observations <rainfall_hourly.json>
```

`fetch`只下載與對齊RD／IR；`rainfall`取得最近1–24小時逐時測站資料並產生RA。完整選項見[使用說明](docs/realtime-plugin.md)。

## 離線示範與測試

單元測試不會訓練模型，也不依賴外網：

```powershell
python -m unittest discover -s tests -v
```

若要使用隨附歷史示範影像，先將 `database.zip` 解壓到專案根目錄得到 `database/`，再執行：

```powershell
python -m typhoon_rainfall predict --config configs/default.json --limit 5
python -m typhoon_rainfall evaluate --config configs/default.json
```

也可在IDE執行 `run_predict_sample.py` 或 `run_evaluate.py`。ZIP是小型示範資料，包含173組訓練清單、55組驗證清單，不是隨附模型使用的完整訓練／測試資料。核心評估入口輸出示範集逐筆類別RMSE，與模型摘要中的完整獨立測試指標需分開閱讀。

離線資料主要結構：

```text
database/
├── train_data_RD/{train,val}/
├── train_data_IR/{train,val}/
├── train_data_RA/EWB01/{train,val}/
├── train_data_GI/{lon.png,lat.png}
└── train_list/t2t+1/{train.txt,val.txt}
```

清單每行為 `source.png;target.png`，例如 `201308271130.png;201308271230.png`，表示來源與目標相差一小時。

## 進階研究：自行訓練

一般使用不需要此步驟。若持有完整歷史資料，可在同一環境執行：

```powershell
python train_realtime.py --dataset-root <完整歷史database目錄> --checkpoint checkpoint/Unet3_retrained.h5
```

這個入口讀取完整資料的 `N-4/t2t+1`、`train_data_rainfallmap/EWB01`、RD／IR／GI及本島遮罩，訓練九通道、101類Unet3。完整資料不隨Git提供，示範ZIP不能代替它。範例使用不同輸出檔名，避免覆蓋固定示範權重。

兩套訓練流程刻意保留：`train_realtime.py` 產生簡化版 Unet3；`run_train.py` 與核心研究CLI保留原始論文研究方法及模型選項，兩者的資料格式、loss與選模方式不同。

核心研究CLI另保留 `train`、`evaluate`、`predict`，共用相同依賴與融合規則。變更模態、類別數或架構時，必須搭配相符訓練權重。

## 文件

- [使用說明](docs/realtime-plugin.md)：操作、時間、輸入輸出與來源。
- [模型與評估摘要](docs/research/multimodal-unet3-2026-11-06.md)：固定模型的訓練、指標與限制。
- [資料介接規格](docs/research/cwa-data-feasibility-2026-10-30.md)、[RD／IR對齊](docs/research/rd-ir-alignment.md)、[RA參數](docs/research/thesis-ra-parameters.md)：目前實作方法。

評估文件供理解模型表現，不是推論執行所需檔案。詳細本機輸出與訓練報告留在被忽略的 `output/`，公開文件只保留必要摘要。
