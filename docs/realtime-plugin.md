# 即時氣象資料與碩論模型外掛

外掛位於 `typhoon_rainfall_realtime/`，使用現有 TensorFlow 2.9.1 環境，透過原本 `RainfallPredictor` 執行單次推論。模型架構、權重、資料融合、訓練及評估程式均不修改。

## 使用方式

在專案根目錄的 Python 3.9 環境安裝外掛需求（包含原需求與 pyproj 3.6.1；原 `requirement.txt` 不變）：

```powershell
python -m pip install -r requirement_realtime.txt
```

接著執行：

```powershell
python run_realtime.py fetch
python run_realtime.py predict
python run_realtime.py rainfall --hours 6
```

直接執行 `run_realtime.py` 預設為 `predict`。也可使用 `python -m typhoon_rainfall_realtime predict`，或執行 `python run_realtime.py --help` 查看指令。

若使用專案 `.venv`：

```powershell
.\.venv\Scripts\python.exe run_realtime.py predict
```

- `fetch`：下載最近可用的 RD／IR，投影至共同網格，保存原圖、模型輸入、定位 metadata 及 `alignment.png` 疊圖；無須 TensorFlow 或權重，需 requests、NumPy、OpenCV、Matplotlib 與 pyproj。
- `predict`：下載、前處理並載入既有權重，輸出原論文的雨量類別 CSV／PNG 和三圖並排的 `overview.png`。
- `rainfall --hours 6`：保存全臺逐時表格、測站 JSON、正規化的近六小時雨量、官方 QZT 參考圖，並以 IDW 產生每個小時的 RA。`--hours` 支援 1–24。加 `--observations-only` 可只匯出數值。
- `ra --observations <rainfall_hourly.json>`：以已保存的逐時數值重新生成 RA，不重新抓測站雨量；若邊界已快取或提供 `--boundary-file`，可離線執行。

每次建立獨立的 `output/realtime/<執行時間>_<指令>/`，包含 `manifest.json`。這個目錄已加入 Git 忽略，避免把即時資料或輸出當成程式提交。`--output-root` 可指定其他輸出根目錄；自訂位置須自行排除版本控制。

```powershell
python run_realtime.py predict --config configs/default.json --max-age-minutes 90 --max-skew-minutes 20
```

預測限定既有 RD＋IR、128×128、100 類、t+1 設定（RA／GI 關閉）。預設讀 `configs/default.json` 的 `Unet3` 與 `checkpoint/Unet3_val_min_RMSE.h5`；相對權重路徑以專案根目錄解析，不會因找不到對應模型就偷偷 fallback 到另一個 `.h5`。自訂設定也必須符合此輸入契約。

## 圖源與前處理

官方公開來源不需要 Cookie 或會員授權碼：

| 輸入 | 官方圖片 | 時間描述 | 新適配方式 |
| --- | --- | --- | --- |
| RD | [O-A0058-003 PNG](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0058-003.png) | [XML](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0058-003.xml) | 等經緯影像，依共同格點逆映射像素 |
| IR | [O-C0042-004 JPG](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-C0042-004.jpg) | [XML](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-C0042-004.xml) | TM投影與經緯格線校正，再映射共同格點 |

限定已校準產品（RD 3600×3600、8-bit RGB PNG；IR 800×800 JPG），不再各自套固定crop。兩者使用共用 `geometry.GridSpec`，約119.3467–122.7600°E、21.8937–25.3086°N的128×128網格；bbox代表像素外邊界、取格點中心、北向上，RA亦使用相同座標。此範圍由176張舊訓練RD格線校準，56張保留驗證最大差異小於0.189個模型像素。

以pyproj作逆投影、8倍細格點bilinear取樣，再用OpenCV `INTER_AREA` 積分縮成128×128。RD保存RGB、IR先轉灰階；正規化與四通道融合仍由核心執行。產品、尺寸、位深、來源覆蓋或可用圖面不符會拒絕處理，避免混入標題／署徽或補黑邊後繼續預測。

`alignment.json`與manifest保存共同網格、各來源CRS／像素轉換／取樣範圍；`grid.npz`保存共同座標。`alignment.png`將輸入及疊圖與舊RA的4700格本島輪廓比較。每次亦保存校準檢核JSON。詳見[RD／IR適配說明](research/rd-ir-alignment.md)。

下載前後核對 metadata，以降低固定最新圖網址在下載途中換檔的風險；保留原圖 SHA-256、觀測／發布／取得時間。預設拒絕觀測超過 90 分鐘、明顯來自未來或 RD／IR 觀測時間相差超過 20 分鐘的資料。可調整允許的時效，但畫面及 manifest 一律保存實際時間。

預報基準為 RD 的觀測時間，目標為其後一小時，並非電腦執行時間後一小時。IR 標示的時刻可能為掃描開始，與 RD 時間語意不同，故另存其時間。輸出維持原論文 0–99 雨量標籤及色階；本次示範沒有新增雨量校正或宣稱新資料準確度。

## 雨量來源

- [全臺最近 24 小時逐時表](https://www.cwa.gov.tw/V8/C/P/Rainfall/MOD/22.html)：每欄各自一小時，按表頭月日和起終時解析。`-` 表示 0 mm；`X` 表示缺值，缺值不補零。
- [O-A0002-001 JSON](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0002-001.json)：補測站位置，選 WGS84 座標；其 Past1hr／3hr 等為最新滑動累積，與逐時表分開保存。不同站碼格式僅在唯一候選時匹配，未匹配保留。
- [QZT 小時累積雨量圖](https://www.cwa.gov.tw/Data/rainfall/QZT.jpg)：供視覺參考，與測站自算的 RA 格網不同。

使用者提供最終論文後，已確認當年的方法就是 IDW，最終 N=5、u=2；原始實作程式尚未找到。目前外掛依已確認的公式與參數補上 RA，距離公式和精確地理定位仍標記為實作選擇，不宣稱完整重現。[論文參數核對](research/thesis-ra-parameters.md)

## RA 圖生成

```powershell
.\.venv\Scripts\python.exe run_realtime.py rainfall --hours 6
.\.venv\Scripts\python.exe run_realtime.py ra --observations output/realtime/<快照目錄>/rainfall_hourly.json
```

流程為「逐小時各站毫米雨量＋WGS84 座標 → 排除離島站及缺值 → Haversine 球面距離 → 最近 5 站、u=2 距離平方反比加權 → 臺灣本島遮罩 → RA」。N=5、u=2 與缺站排除方式由最終論文印刷第29頁確認。預設不加搜尋半徑限制，論文未記載此參數；`--radius-km` 是可選的額外截斷。Haversine 是目前明確採用的距離公式，論文只寫由經緯度計算距離，未指明公式。同址站使用有效站平均；有效零雨量保留。

論文採臺灣本島、128×128格網，約3km／格，排除離島。外掛取本島縣市的主要陸地組件，測站亦依此篩選；即時站數依現行有效資料，並非假裝沿用論文歷史556站。RA預設與RD／IR共用上面校準的 `geometry.GridSpec`。`--grid-size 256` 或 `512` 是額外展示選項，不會改動128×128模型輸入。

海陸遮罩取自[桃園市政府 GIS 提供的 NLSC 縣市界](https://trms.tycg.gov.tw/server/rest/services/Hosted/COUNTY_MOI_1090820/FeatureServer/0)，原始資料版本 COUNTY_MOI_1090820，下載時轉 WGS84 並簡化 0.003°，來源與指紋保存在 manifest。[NLSC 資料集與授權](https://data.gov.tw/dataset/7442)。來源包含22縣市；外掛排除澎湖、金門、連江，並以各本島縣市最大面組件排除蘭嶼、龜山島等離島。這是新的現行邊界遮罩，不是論文原始遮罩。第一次取得後快取於 `.cache/realtime-boundary/`；也可用 `--boundary-file` 指定本地 GeoJSON。

每個小時的 `ra/RA_<時間與時區>/` 包含：

- `RA_mm.csv`：浮點累積雨量 mm；海面與無覆蓋區為 NaN。
- `RA_labels.png`：依論文 EWB01 每1mm四捨五入（0.5向上）、上限100，101類；本島外背景0。若明確加半徑導致本島無覆蓋則用255表示缺值，並保留遮罩。
- `valid_mask.png`：255 為有效雨量，0 為海面或無覆蓋；零雨量不等於缺值。
- `RA_color.png`：128×128（或指定尺寸）的彩色雨量圖。
- `RA_map.png`：包含縣市邊界、毫米色階、實際觀測區間與 IDW 標示的地圖。
- `metadata.json`：該小時時間與覆蓋統計；共用 `grid.npz` 保存逐格座標與陸地遮罩，`ra_manifest.json` 保存方法、來源及設定。

RA 是觀測內插結果，不是模型預測。既有 Unet3 權重仍只接受 RD＋IR 四通道、100類，與論文101類有差異；不會把新 RA 自動加進輸入或改動核心類別數。100與255也不可直接交給核心舊 label loader（其會截成99）；未來接入RA模型前需先處理遮罩並確認格網、標籤和權重契約。

算法依據為使用者提供的論文4.5節；補充參考：[IDW 距離與權重定義（Esri 官方）](https://pro.arcgis.com/en/pro-app/3.6/tool-reference/spatial-analyst/how-idw-works.htm)。目前僅還原已確認的部分，仍需原始程式／格網才能證明完整重現。

## 驗證

```powershell
python -m unittest discover -s tests -v
python run_realtime.py fetch
python run_realtime.py rainfall --hours 3
python run_realtime.py predict
```

單元測試使用離線樣本；上面最後三項需外網，`predict` 另需 TensorFlow 2.9.1 和現有權重。若失敗，CLI 回傳非零狀態，已建立的輸出資料夾會保存失敗 manifest，不能把 `fetch` 成功誤認為模型推論成功。公開圖源和官網 HTML 可能異動，供應變化時優先修正外掛來源解析，不必改動核心模型。

### 2026-10-31 本機驗證紀錄

- 使用專案隔離 `.venv`：Python 3.9.25、TensorFlow 2.9.1，原 `requirement.txt` 不變；CPU 推論。
- 45 項單元測試通過。既有 checkpoint 測試只修正為跨平台 Path 比較，核心權重選擇程式未改。
- `fetch` 實際匿名抓圖成功；`predict` 已載入原 Unet3 權重、完成 128×128×4 輸入推論並輸出 CSV、PNG、overview 及時間 manifest。
- 首次推論 RD 基準 08:40、IR 08:30，目標 09:40（臺灣時間）；輸出為 128×128，全部是類別 0。這是實際模型輸出，不代表已通過近期天氣的精度驗證。
- `rainfall --hours 6` 實際匯出 1,340 測站、8,040 筆逐時資料；保留 8 筆缺值，座標皆匹配，QZT 參考圖片取得成功。
- Matplotlib 快取預設使用專案 `.cache/matplotlib`；Windows 沙箱若不能使用系統 TEMP，可在測試前將 TEMP／TMP 指向專案 `.cache/test-tmp`。
