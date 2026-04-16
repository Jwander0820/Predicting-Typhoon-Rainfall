"""PyCharm-friendly entrypoint: evaluate the validation split.

直接在 PyCharm 按 RUN 會跑完整驗證清單，輸出預測結果與 RMSE 摘要。
若只是想快速確認流程，建議先跑 `run_predict_sample.py`。

模型、checkpoint、輸入模態都由 `configs/default.json` 決定。
執行時 console 會列出 Run context 與 Model loading。
"""

from typhoon_rainfall.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/default.json", "evaluate"]))
