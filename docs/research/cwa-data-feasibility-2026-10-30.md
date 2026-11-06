# 最新氣象影像套用既有碩論模型：資料介接可行性研究

研究日期：2026-10-30（Asia/Taipei）。專案盤點基準：`ea0ef60`。本輪僅研究並新增本筆記，未修改程式、設定、模型、原始資料或 Git 暫存區。

## 1. 本階段目標與結論

使用者確認的目標是：取得最近可用的紅外線與雷達影像，套用既有碩論模型，輸出下一小時的雨量分布預測圖，用來展示碩論專案可接收現行資料並執行推論。

這條流程可行，且專案已有相應入口與權重。近期颱風回測、擴充資料、更新模型及全面準確度驗證，均屬後續擴充，**不是本次 demo 的前置條件**。本階段也不需要先取得多年的歷史影像、雨量標籤或改成數值氣象模型。

本輪查到最接近舊 crawler 的現行官方產品是：

| 原有輸入 | 優先採用的官方產品 | 實際尺寸與內容 | 本次確認 |
| --- | --- | --- | --- |
| RD：`CV1_TW_3600` 雷達回波圖 | `O-A0058-003`，鄰近區域、無地形 | 3600×3600 RGB PNG；含海岸、行政界線及灰底 | 檔頭讀取成功，並下載一張至系統暫存檢視；原命名在 `cwa.gov.tw` 網域仍可取得 |
| IR：`TWI_IR1_Gray_800` 黑白紅外線雲圖 | `O-C0042-004`，臺灣高解析黑白雲圖 | 800×800 JPG；灰階雲圖內容，完整 JPG 含彩色署徽 | 檔頭讀取成功，並下載一張至系統暫存檢視；原命名在 `cwa.gov.tw` 網域仍可取得 |

產品分類來源：[雷達官方說明](https://www.cwa.gov.tw/Data/data_catalog/3-6-1.pdf)、[紅外線官方說明](https://www.cwa.gov.tw/Data/data_catalog/3-7-2.pdf)。尺寸來自本次實際 PNG IHDR／JPEG SOF 抽樣；官方 metadata 顯示兩類產品均每 10 分鐘更新：[RD metadata](https://opendata.cwa.gov.tw/webapi/datasetMetadata/O-A0058-003)、[IR metadata](https://opendata.cwa.gov.tw/webapi/datasetMetadata/O-C0042-004)。

## 2. 專案已經具備的推論流程

下列是本地原始碼與檔案的唯讀盤點結果，未執行模型。

| 項目 | 現況與證據 |
| --- | --- |
| 即時單次推論入口 | [predict_1hour_rainfall.py](../../predict_1hour_rainfall.py) 已有「下載 RD／IR → 建立模型 → 推論 → 存檔」流程 |
| 抓圖模組 | [crawler.py](../../typhoon_rainfall/experimental/crawler.py) 仍組合舊 `www.cwb.gov.tw` 圖片網址，並使用舊時間推算與固定像素裁切 |
| 模型及權重 | 單次推論明確使用 `Unet3`；本地存在 `checkpoint/Unet3_val_min_RMSE.h5`，9,969,704 bytes，尚未驗證本機載入 |
| 模型輸入 | RD 三通道 + IR 單通道，合成 `128×128×4`；兩者正規化除以 255，排列為 RD、IR。見 [dataset.py](../../typhoon_rainfall/data/dataset.py) |
| 目前不需補的資料 | 該入口 `use_ra=False`、`use_gi=False`；所以單次 demo 不需 RA 雨量標籤、GI 經緯度圖，也不需先解壓 `database.zip` |
| 輸出 | 由 softmax `argmax` 取得雨量類別，存整數 CSV 及彩色 PNG。見 [predictor.py](../../typhoon_rainfall/inference/predictor.py)、[plotting.py](../../typhoon_rainfall/visualization/plotting.py) |
| 相容環境 | [requirement.txt](../../requirement.txt) 固定 TensorFlow 2.9.1；README 建議 Python 3.9。demo 應先沿用此環境與既有權重 |

因此，下一輪實作重點應是接回最新官方圖片、核對原有前處理，再完成一次既有模型推論。

## 3. 可使用的官方圖源與實測證據

### 官方原命名圖片

下列是本次確認 HTTP 200 並實際檢視的具體樣本，時間為臺灣時間；它們是查核當下的樣本，不是永遠指向最新資料的網址。

- RD：[`CV1_TW_3600_202610301100.png`](https://www.cwa.gov.tw/Data/radar/CV1_TW_3600_202610301100.png)，2026-10-30 11:00。
- IR：[`TWI_IR1_Gray_800-2026-10-30-10-50.jpg`](https://www.cwa.gov.tw/Data/satellite/TWI_IR1_Gray_800/TWI_IR1_Gray_800-2026-10-30-10-50.jpg)，圖面標示開始觀測 10:50、結束觀測 11:00。

官方目前雷達時間清單將 `Tab0/Area1/size0` 對應到 `CV1_TW_3600`；選項程式將 `Tab0` 定義為無地形、`Area1` 定義為臺灣鄰近區域，與 `O-A0058-003` 相符。[官方時間清單](https://www.cwa.gov.tw/Data/js/obs_img/Observe_radar.js)、[官方選項程式](https://www.cwa.gov.tw/V8/assets/js/observe/function-radar.js?v=20220628)

這使得更新網域並沿用原圖種，成為最小修改的候選路線；但舊 crawler 每 30 分鐘回退推算網址的規則仍應改以實際可用圖片時間為準。

### 公開最新檔與時間描述

| 產品 | 圖片入口 | 容器／時間資訊 | 本次抽樣時間 |
| --- | --- | --- | --- |
| RD | [O-A0058-003.png](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0058-003.png) | [O-A0058-003.xml](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0058-003.xml) | `DateTime=2026-10-30T11:00:00+08:00`；`sent=11:06:41+08:00` |
| IR | [O-C0042-004.jpg](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-C0042-004.jpg) | [O-C0042-004.xml](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-C0042-004.xml) | `Datetime=2026-10-30T10:50:00+08:00`；`sent=11:05:40+08:00` |

這些公開圖片及描述可供 demo 圖源查核；固定名稱入口的內容會更新，實作時須保存該次使用的原圖及時間。尚未確認其有穩定性承諾，長期維護應跟隨官方產品與服務公告。

若採會員檔案 API，官方格式為：

```text
https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/{dataid}?Authorization={CWA_API_KEY}&format=XML
```

本輪未使用使用者授權碼；對不附授權碼的 `O-A0059-001` fileapi 請求實測為 HTTP 401。**公開圖片已可讀取，會員 API 申請不是目前圖源可行性的阻塞條件。** 會員介接方式與 ETag 快取依[官方開發指南](https://opendata.cwa.gov.tw/devManual/description)辦理；占位符不是實際金鑰。

## 4. 沿用原模型時，需做的有限確認

### 裁切與通道

原 crawler 使用：

| 圖種 | 原圖裁切（Python 切片） | 後續處理 |
| --- | --- | --- |
| RD | `image[716:2762, 830:2876]`，2046×2046 | 縮放到 128×128、保存彩色 PNG |
| IR | `image[261:558, 247:544]`，297×297 | 先讀成灰階，再縮放到 128×128 |

本次新圖片尺寸都容納上述範圍，視覺檢視也涵蓋臺灣，適合用原裁切作為初次 demo 候選。下一輪只需把裁切後影像與 repo 的舊 RD／IR 範例核對一次，確認臺灣位置、方向、雷達色階、灰階內容及最終通道數。**目前尚未宣稱已證明與舊訓練圖逐像素地理一致。**

應選 `O-A0058-003` 對應原 RD；`O-A0058-001` 雖然同為 3600×3600，實際是較大範圍，不能僅因尺寸相同沿用裁切。透明圖層及有地形產品也會改變原模型看到的輸入外觀。[雷達產品分類](https://www.cwa.gov.tw/Data/data_catalog/3-6-1.pdf)

IR 應選黑白產品並在前處理後保持單通道；完整 JPG 是三分量編碼、署徽有顏色，不應把整張 JPG 未轉灰階就當作原本的 IR 單通道。彩色、色調強化雲圖與數值亮溫並非這次既有模型需要的輸入。[IR 官方說明](https://www.cwa.gov.tw/Data/data_catalog/3-7-2.pdf)

### 時間與輸出說明

展示結果可標示「RD 資料時間／IR 觀測期間 → 原碩論模型 t+1 小時雨量預測」。保留圖片的實際時間，避免以執行電腦的時間替代圖資時間。

IR 的容器時間在本次樣本對應開始觀測時間；10:50–11:00 的 IR 與 11:00 雷達，不能只因兩個字串不同就判定不能配對。下一輪應明確記錄兩者時間語意，並沿用原研究的配對方式。

模型維持現有 100 類雨量標籤及色階即可。原程式將大於等於 100 的 target label 截為 99，推論輸出也為 0–99；本次先展示原模型輸出，相關標籤尺度改善可留待後續。

## 5. 下一輪最小實作與驗收範圍

建議流程：

```text
最近可用的官方 RD／IR 圖
  → 保存原圖及實際觀測時間
  → 原研究裁切、128×128、RD RGB + IR 灰階、正規化
  → 現有 Unet3 + Unet3_val_min_RMSE.h5
  → 輸出原模型 CSV／預測 PNG，以及輸入圖供展示
```

實作以現有 [crawler.py](../../typhoon_rainfall/experimental/crawler.py) 及 [predict_1hour_rainfall.py](../../predict_1hour_rainfall.py) 為主要接點，視實際圖源調整抓圖、時間及必要的前處理。模型架構、權重與訓練流程沿用。

這階段完成的判準：成功取得一組最近的官方圖片，轉成符合既有模型的四通道輸入，載入現有權重，跑出一組 CSV 和 PNG，並能看出使用的資料時間。這足以展示端到端推論流程；近期颱風回測、精度比較、資料擴充與重新訓練，依使用者指示留到後續。

本輪已完成圖源及檔案尺寸／內容查核、原始碼盤點；**尚未修改抓圖程式、執行模型、驗證權重載入或產出新的預測圖**。本輪下載的圖只放在系統暫存，未寫入 repo 的 `img` 或覆蓋既有範例。

## 6. 補查：RA 雨量分布圖、近幾小時測站雨量與免登入下載

使用者進一步確認：希望取得各地前幾小時降雨資料，以當年的測站雨量空間內插算法重建 RA；並詢問 RD／IR 是否必須登入。以下仍為研究結果，未實作下載器、內插或模型推論。

### 最新雨量圖與全臺逐時數值皆可取得

| 需求 | 免登入官方來源 | 本次確認 |
| --- | --- | --- |
| 已繪製的小時累積雨量分布圖 | [QZT.jpg](https://www.cwa.gov.tw/Data/rainfall/QZT.jpg)、[官方檢視頁](https://www.cwa.gov.tw/V8/C/P/Rainfall/Rainfall_QZT.html) | 無 cookie／授權碼的 Range GET 回 206；[圖資時間清單](https://www.cwa.gov.tw/Data/js/rainfall/RainfallImg_Hour.js)列近兩日、半小時終時的圖。本次樣本為 [2026-10-30 11:30 圖](https://www.cwa.gov.tw/Data/rainfall/2026-10-30_1130.QZT8.jpg) |
| 全臺各測站最近 24 小時、每小時各別雨量 | [全站原始表格 MOD/22.html](https://www.cwa.gov.tw/V8/C/P/Rainfall/MOD/22.html)、[正常檢視頁](https://www.cwa.gov.tw/V8/C/P/Rainfall/Rainfall_Hour.html?ID=22) | 無 cookie／授權碼，HTTP 200；本次 1,329 站，產製時間 10/30 11:36:02，逐時欄為 10→11、09→10、08→09 等，共 24 欄；來源為 HTML，並非 JSON |
| 測站位置、最新雨量累積窗口 | [O-A0002-001.json](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0002-001.json)、[metadata](https://opendata.cwa.gov.tw/webapi/datasetMetadata/O-A0002-001) | 無 cookie／授權碼，HTTP 200；本次 1,328 站，觀測時間 2026-10-30T11:50:00+08:00、sent=11:55:14+08:00；含站碼、TWD67／WGS84 座標及 Past10Min、Past1hr、Past3hr、Past6hr、Past12hr、Past24hr；Now 是當日累積 |

全站逐時表可提供「10:00–11:00、09:00–10:00、08:00–09:00 各下多少雨」。JSON 的 Past3hr 等欄位則是截至觀測時間的累積窗口，不能僅靠一份最新 JSON 就拆出前三小時各別雨量。整點逐時表也不能直接當成 10:30–11:30 的滑動一小時雨量。

可用全站逐時表取雨量、以站點 JSON 補 WGS84 經緯度，再交給原 RA 算法。站碼格式需映射：本次臺北表格站碼為 46692，JSON 為 StationId=466920，WGS84 為 25.037658°N、121.514853°E；不能未檢查就對所有站盲目截碼。兩份資料站數不同，也需保留未匹配狀況。官方表格註記「-」為 0 mm、「X」為無資料。[官方表格說明](https://www.cwa.gov.tw/V8/C/P/Rainfall/Rainfall_Hour.html?ID=22)

單站也有公開 [Plot_Hr/46692.js](https://www.cwa.gov.tw/Data/js/rainfall/Plot_Hr/46692.js)，本次約 4KB，含 Rain_Data 24 點及 Rain_Data_72 72 點；全臺 demo 優先一次讀全站表，無須逐站抓取。該 JS 的 Date.UTC 與圖表時區設定應一併核對，避免把臺灣時間再錯加八小時。

本次未確認 QZT 的獨立 OpenData 代碼。O-A0040-002／004 是日累積產品，不能因為更新頻率為一小時或半小時，就當作過去一小時雨量；O-B0045-001 則是雷達與測站校正的 QPE，與自行內插 RA 仍有資料來源和算法差異。

### 原本測站內插算法的本地盤點

已搜尋目前全部程式與 notebook（含被忽略路徑），並檢查本地完整、非 shallow 的可見 Git 歷史：378b6c2、107943d、3946a17、1d83c68、ea0ef60；舊 train.ipynb／example.ipynb 亦讀取原始 cell source。另確認 database.zip 中沒有前處理程式或 notebook。

目前找到的是預先生成 RA 的載入、標籤轉換、預測值與真值 CSV 的評估，以及陣列繪圖；尚未找到「測站座標＋測站雨量 → 空間內插 → 全臺 RA 格網」的實作。[現有 RA 載入](../../typhoon_rainfall/data/dataset.py)、[現有 CSV／陣列繪圖](../../typhoon_rainfall/visualization/plotting.py)

因此，數值來源可取得，但原算法目前尚未在此 repo 定位到，可能保存在當年的另一份資料前處理腳本。若要重現原論文 RA，需沿用原算法、區域、格網、遮罩及標籤轉換，而非直接把官方彩色 JPG 當成單通道 RA 標籤。

### 圖片直接下載不需會員登入

本次再次以不帶 cookie／授權碼的 GET Range 查驗下列圖片，兩者均回 HTTP 206：

- [RD PNG：O-A0058-003](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0058-003.png)
- [IR JPG：O-C0042-004](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-C0042-004.jpg)

上表 QZT 圖、全站逐時表及 O-A0002-001 JSON 也已匿名取得。資料集介紹頁／會員下載功能與公開檔案網址要分開看；正式 datastore／fileapi／historyapi 等服務依官方指南使用會員授權碼，程式取得授權碼後可自動介接，不需要每次手動登入網頁。

O-A0002-001 當前 metadata 列有 historyAPI，但不帶授權碼實測 401，保留範圍尚未驗證。本次以公開逐時表已能取得近 24 小時各站數值。舊 O-A0002-002 的 metadata 本次回 success:false、公開 S3 JSON／XML 均 404，因此不能依舊說明文件推薦它作為現行可用資料來源。

## 附錄：已查到但本次 demo 不需要的資料

以下僅保留供未來回測或更新模型參考，不列入本階段實作。

| 資料 | 官方代碼與來源 | 未來可能用途 |
| --- | --- | --- |
| 數值雷達 | [O-A0059-001](https://opendata.cwa.gov.tw/webapi/datasetMetadata/O-A0059-001) | dBZ 數值、921×881、0.0125°；若改用數值輸入，須另行處理模型契約 |
| 過去 1 小時降雨估計 QPE | [O-B0045-001](https://opendata.cwa.gov.tw/webapi/datasetMetadata/O-B0045-001) | 441×561、0.0125°、mm；可作後續雨量目標或比較資料，屬估計值 |
| 未來 1 小時降雨預報 QPF | [F-B0046-001](https://opendata.cwa.gov.tw/webapi/datasetMetadata/F-B0046-001) | 官方預報基準，不能當作觀測真值 |
| 歷史測站雨量 | [CODiS](https://codis.cwa.gov.tw/StationData) | 後續近期事件回測、測站交叉比較 |

本次對數值雷達、QPE、QPF 的官方公開 XML 各抽樣 8,192 bytes，均回傳 HTTP 206，確認真實檔頭與時間。樣本為 `2026-10-30T10:50:00+08:00`；QPE/QPF `sent=11:05:18+08:00`，所以 10 分鐘更新不等於沒有發布延遲。[雷達樣本](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0059-001.xml)、[QPE 樣本](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-B0045-001.xml)、[QPF 樣本](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Forecast/F-B0046-001.xml)

QPF 樣本結構化起點為 118°E/20°N，文字描述則為 117.975°E/19.975°N，相差兩個格距；日後若進行逐格比較再查明。本次沿用影像模型不會用到這份數值格網。

官方有短期歷史 API，可依指南列支援資料集及供應期間；本輪未用會員授權碼查明 QPE/QPF/IR 的歷史支援、保留範圍與配額。需要多年資料時可再查 [CWA 資料申購](https://hdps.cwa.gov.tw/hdps/)、[衛星產品整合系統](https://spd.cwa.gov.tw/)及官方產品申請途徑。[歷史 API 指南](https://opendata.cwa.gov.tw/devManual/description)

本地 `database.zip` 共有 173 組訓練配對、55 組驗證配對，所有配對時間差為 60 分鐘；訓練來源涵蓋 2013-08、2015-09、2016-09，驗證為 2016-09-26 至 28。README 說明為三場訓練颱風、一場驗證颱風。目前 [evaluation/runner.py](../../typhoon_rainfall/evaluation/runner.py) 算的是 label RMSE，未使用真值 CSV；日後做近期颱風回測時再處理實際毫米雨量及評估定義。
