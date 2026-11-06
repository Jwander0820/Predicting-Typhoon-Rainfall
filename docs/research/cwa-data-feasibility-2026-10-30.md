# 氣象資料介接與模型輸入規格

更新至2026-11-07。本文說明目前已接通的來源與九通道流程；操作見[使用說明](../realtime-plugin.md)。檔名保留原研究日期以維持文件連結。

## 現行流程

```text
最近完整小時測站雨量 → IDW RA
同小時中點 RD／IR → 共同128×128網格
原始 LON／LAT GI → 九通道融合
Unet3_val_min_RMSE.h5 → 下一小時雨量分布
```

隨附模型為Unet3、九通道、101類，已完成訓練與歷史獨立測試。一般使用直接執行 `python run_realtime.py`，安裝只需同一份 `requirement.txt`，不需要先訓練。

## 資料來源

| 資料 | 現行來源 | 處理 |
| --- | --- | --- |
| RD | CV1_TW_3600，對應O-A0058-003 | 3600×3600 RGB，等經緯逆映射 |
| IR | TWI_IR1_Gray_800，對應O-C0042-004 | 800×800 JPEG，TM校正後轉灰階 |
| 逐時雨量 | [最近24小時逐時表](https://www.cwa.gov.tw/V8/C/P/Rainfall/MOD/22.html) | 各站各小時獨立累積，保留缺值 |
| 測站座標 | [O-A0002-001 JSON](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0002-001.json) | 唯一站碼匹配、WGS84 |
| GI | ZIP中的原始兩張灰階圖 | 原位元組複製、除以255 |

完整預測使用指定時刻檔名：觀測RA例如17:00–18:00時，RD／IR取17:30，預測18:00–19:00。時間由官方檔名聲明；發布時間未知時不以取得時間冒充。官方[雷達清單](https://www.cwa.gov.tw/Data/js/obs_img/Observe_radar.js)及[衛星清單](https://www.cwa.gov.tw/Data/js/obs_img/Observe_sat.js)提供圖種與時間。

公開影像與測站資料不需要會員Cookie或API授權碼。會員FileAPI／HistoryAPI是不同介面，本流程不依賴它們。

`fetch`另使用公開最新 [RD PNG](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0058-003.png)／[XML](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0058-003.xml)、[IR JPG](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-C0042-004.jpg)／[XML](https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-C0042-004.xml)。固定網址會換檔，下載前後核對metadata，保存觀測時間與原圖SHA；它們不是歷史資料庫。

## RA與GI

RA已納入預設輸入。外掛由本島有效測站，以最近5站、u=2的IDW產生128×128雨量場，編成0–100標籤；核心複製為三通道、除以101。缺站不補零，海面由遮罩處理。[RA方法](thesis-ra-parameters.md)

GI也已接入。LON向東增加、LAT向南減少，皆為128×128、0–254的原始灰階圖，與訓練資產指紋相同，提供相對位置。灰階值不能直接當作實際經緯度。

JSON內Past1hr／Past3hr等滑動累積不能拆成各小時真值；本流程以逐時HTML作RA來源。QZT只作視覺參考，不直接取代自算RA。

## 資料契約與限制

- 共同格網與取樣方式見[RD／IR對齊](rd-ir-alignment.md)。
- 通道依序為RD RGB、IR、RA×3、LON、LAT；尺寸128×128×9。
- 輸出101個1mm類別，100表示100mm以上，不是連續精確雨量。
- 拒絕格式／尺寸／覆蓋不符、未完成觀測、過期來源與已結束預報。
- 保存成功／失敗manifest及來源時間，完整輸出留在被Git忽略的 `output/`。

模型與規則固定，供論文簡易復現與多模態示範。[模型摘要](multimodal-unet3-2026-11-06.md)保留測試結果與限制。共同方法可重建，不代表恢復全部歷史原始資料或已證明現行圖源的預報精度。
