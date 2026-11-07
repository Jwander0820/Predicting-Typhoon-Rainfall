# 即時多模態降雨預測

目前入口為 `run_realtime.py`，以RD／IR／RA／GI九通道輸入，載入隨附Unet3權重預測下一個逐時雨量區間。一般使用不需要先訓練。

## 執行

所有流程在同一個Python3.9環境使用同一份依賴：

```powershell
python -m pip install -r requirement.txt
python run_realtime.py
```

也可執行 `python -m typhoon_rainfall_realtime predict`。CPU可以推論；GPU只提供可選加速，不需要另一份環境需求。

| 項目 | 值 |
| --- | --- |
| 設定 | `configs/realtime.json`；`configs/default.json`亦為相同模型契約 |
| 權重 | `checkpoint/Unet3_val_min_RMSE.h5` |
| 模型資訊 | `checkpoint/Unet3_val_min_RMSE.metadata.json` |
| 輸入 | 128×128×9，RD RGB／IR／RA×3／LON／LAT |
| 輸出 | EWB01 101類，0–99及100mm以上 |
| 正規化 | RD／IR／GI除以255；RA除以101 |

下載前檢查權重輸入通道與輸出類別；有metadata時也核對權重SHA256。單純更改模態旗標不會轉換既有權重。

## 資料與時間配對

1. 讀取最近完整小時的各站雨量及WGS84座標，保留缺值狀態。
2. 篩選本島有效測站，以IDW最近5站、u=2產生RA。
3. 下載該小時中點的指定時刻RD／IR，映射共同網格。
4. 從隨附 `database.zip` 讀取原始GI兩張灰階圖；即時流程不需解壓ZIP。
5. 依核心融合規則建立九通道輸入，載入固定權重並輸出結果。

例如觀測RA為17:00–18:00，RD／IR取17:30，預測18:00–19:00。指定時刻影像的觀測時間來自官方檔名，發布時間未知時保留空值。

來源發布有延遲，預報區間可能已開始。`manifest.json`記錄觀測區間、來源中點、預報區間、產出時刻與目標時段已過分鐘；不是以執行時刻加一小時。程式拒絕未完成觀測、過期來源與已結束預報，完成推論後也再檢查一次。

## 輸出

每次建立 `output/realtime/<執行時間>_predict/`，不覆蓋前次結果。預設成功後只保留 `forecast.png`、`prediction_mm.csv`、摘要 `manifest.json`、`inputs/`、`alignment.png` 與 `observed_ra.png`。後者是 IDW 觀測 RA 的彩色毫米圖，並非送入模型的標籤圖，也不是官方雨量圖。

使用 `python run_realtime.py --detailed-output` 保留下列完整快照。失敗執行仍保留診斷資料；不清除任何先前執行目錄。

| 檔案／目錄 | 用途 |
| --- | --- |
| `forecast.png` | 前一小時觀測／下一小時預測對照，主要結果圖 |
| `prediction_mm.csv` | 整數毫米預測；海面NaN，100表示100mm以上 |
| `prediction.csv`、`prediction.png` | 類別值及彩色圖 |
| `manifest.json` | 來源、時間、模型指紋與狀態 |
| `inputs/` | 真正輸入模型的RD、IR、RA、LON、LAT |
| `raw/`、`observations/` | 原始圖片、測站表格與正規化雨量 |
| `ra/` | 觀測RA數值、標籤、遮罩、地圖及方法紀錄 |
| `alignment.png`、`alignment.json`、`grid.npz` | 定位及對齊檢核 |

`alignment.png`紫色線來自原研究4700格本島輪廓，只供目視比對，沒有畫入 `inputs/`或模型特徵。GI是LON／LAT兩張原始圖。

整個 `output/`被Git忽略，其內容不是一般使用的前置條件。若以 `--output-root`指定其他位置，需自行決定是否納入版本控制。

## 其他指令

```powershell
python run_realtime.py --help
python run_realtime.py fetch
python run_realtime.py rainfall --hours 6
python run_realtime.py rainfall --hours 6 --observations-only
python run_realtime.py ra --observations <rainfall_hourly.json>
```

- `fetch`：取得最新RD／IR並對齊，不載入模型，不產生預報。
- `rainfall`：取1–24小時逐時雨量並生成RA；`--observations-only`只匯出觀測數值。
- `ra`：由保存的逐時雨量重新產圖，不重新下載測站雨量。邊界已快取或傳入 `--boundary-file`時可以離線執行。
- `predict`：與不帶參數相同，可加 `--config`、`--geography-source`、`--boundary-file`或 `--output-root`。

失敗時回傳非零狀態，已建立的執行目錄保留失敗manifest；取得圖資與完成預測是不同狀態。

## 來源與前處理

| 資料 | 來源與用途 |
| --- | --- |
| RD | `CV1_TW_3600_YYYYMMDDHHMM.png`，對應O-A0058-003，3600×3600、8-bit RGB PNG |
| IR | `TWI_IR1_Gray_800-YYYY-MM-DD-HH-MM.jpg`，對應O-C0042-004，800×800 JPEG，轉灰階 |
| 逐時雨量 | [最近24小時逐時表](https://www.cwa.gov.tw/V8/C/P/Rainfall/MOD/22.html)，各欄一小時 |
| 測站座標 | [O-A0002-001 JSON](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0002-001.json)，只用WGS84 |
| GI | ZIP內的 `database/train_data_GI/lon.png`、`lat.png` |

公開圖源不需要會員Cookie或API授權碼。指定時刻可由官方[雷達清單](https://www.cwa.gov.tw/Data/js/obs_img/Observe_radar.js)與[衛星清單](https://www.cwa.gov.tw/Data/js/obs_img/Observe_sat.js)核對。`fetch`使用公開最新S3物件與XML，下載前後核對metadata並檢查時效；完整預測使用指定小時圖，不以較新的latest圖替代。

RD／IR／RA共用約119.3467–122.7600°E、21.8937–25.3086°N的128×128格網。RD依等經緯定位，IR依TM與經緯格線校正，8倍細格bilinear取樣後作AREA平均。格式、尺寸、位深或覆蓋不符會拒絕處理。[對齊方法](research/rd-ir-alignment.md)

雨量表中 `-`表示0mm，`X`表示缺值，不補零；站碼只在唯一匹配時結合座標。缺站與本島外站不參與IDW。Haversine為目前距離公式，預設最近5站、u=2、無搜尋半徑；RA採half-up四捨五入截至100。本島外背景0，毫米CSV海面為NaN；完整預測拒絕本島缺值或255標籤。[RA參數](research/thesis-ra-parameters.md)

本島邊界取自[桃園市政府提供的NLSC縣市界](https://trms.tycg.gov.tw/server/rest/services/Hosted/COUNTY_MOI_1090820/FeatureServer/0)，版本COUNTY_MOI_1090820，轉WGS84並簡化0.003°；來源指紋寫入manifest，[資料與授權](https://data.gov.tw/dataset/7442)。第一次下載後快取於 `.cache/realtime-boundary/`，它與歷史4700格遮罩有差異。

官方QZT雨量圖只作參考，不直接當模型RA；逐時表與JSON內Past1hr／Past3hr滑動累積分開保存。

## 固定權重與可選訓練

正常預測不會呼叫訓練程式。權重指紋、訓練與主要指標見[模型摘要](research/multimodal-unet3-2026-11-06.md)。評估文件只供理解與驗證模型，不是執行依賴。

需要研究不同設定時，另備完整歷史資料，仍在相同環境執行：

```powershell
python train_realtime.py --dataset-root <完整歷史database目錄> --checkpoint checkpoint/Unet3_retrained.h5
```

完整資料不隨Git提供，示範ZIP不能代替完整訓練集。訓練入口讀取 `N-4/t2t+1`、`train_data_rainfallmap/EWB01`、RD／IR／GI與本島遮罩，記錄時間檢查、資料指紋與最佳驗證權重。

## 目前限制

這版以論文概念一致為範圍，不要求逐像素重現歷史測站、海岸或渲染。歷史獨立測試RMSE與持續性預報接近，高雨量低估；新圖源與現行天氣準確度尚需事後觀測驗證。公開圖源格式變更時，資料介接仍可能需要維護，無須因此自動重訓模型。
