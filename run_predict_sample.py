"""PyCharm-friendly entrypoint: run a small validation prediction sample.

直接在 PyCharm 按 RUN 這個檔案即可，不需要另外填 Parameters。
它只推論前 25 筆驗證資料，適合作為第一次確認環境與模型載入的smoke test。

基礎參數來自 `configs/default.json`，使用隨附的多模態 Unet3 權重：

- model.name: `Unet3`
- model.num_classes: `101`
- data input modes: `RD + IR + RA + GI`，共九通道
- checkpoint: `checkpoint/Unet3_val_min_RMSE.h5`

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
                "--limit",
                "25",
            ]
        )
    )
