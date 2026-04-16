"""PyCharm-friendly entrypoint: train with the default research config.

這會真的開始訓練模型，執行時間會比推論與驗證久。第一次測試環境時不建議先跑本檔案。

訓練參數來自 `configs/default.json`，包含模型名稱、epoch、batch size、
learning rate、loss 是否使用 focal loss、checkpoint 輸出位置等。
執行時 console 會先列出 Run context，方便確認本次訓練設定。
"""

from typhoon_rainfall.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/default.json", "train"]))
