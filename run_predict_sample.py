"""PyCharm-friendly entrypoint: run a small validation prediction sample.

直接在 PyCharm 按 RUN 這個檔案即可，不需要另外填 Parameters。
它只推論前 25 筆驗證資料，適合作為第一次確認環境與模型載入的smoke test。

基礎參數來自 `configs/default.json`，但本檔案額外用 CLI override
將模型改成 `Unet3`，方便測試既有的 Unet3 checkpoint：

- model.name: 本檔案覆蓋為 `Unet3`
- model.num_classes: 預設 `100`
- data input modes: 預設 `RD + IR`
- checkpoint: 預設自動尋找 `checkpoint/Unet3_val_min_RMSE.h5`

執行時 console 會印出 Run context 與 Model loading，可用來確認
本次到底調用了哪個模型、checkpoint、資料 split 與輸出位置。
"""

from typhoon_rainfall.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--config",
                "configs/default.json",
                "predict",
                "--model-name",
                "Unet3",
                "--limit",
                "25",
            ]
        )
    )
