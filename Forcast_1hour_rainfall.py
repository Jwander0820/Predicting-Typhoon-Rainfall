# -*- coding: utf-8 -*-
from core.get_data import WeatherCrawler
from core.plotting import Plotting
from core.predictor import Predictor

from utils.loss import *  # 調用自訂義的損失函數
from utils.metrics import *  # 調用自訂義的評價函數


def main():
    # =============================================================================
    # 接下來將爬取到的資料丟到模型中，產生預測資料
    # 預訓練的模型為RD+IR預測未來一小的案例，並放在checkpoint資料夾中
    # 注意:因為訓練的資料(颱風資料)和爬蟲下來的一般雷達回波圖和紅外線雲圖有差異 (淺藍底vs白底、不同的框線)
    # 所以導致預測的結果為空白的情況，也可能因為模型較淺層較小型所以預測表現自然也不佳
    # 若要改善一般預測的情況可能需要重新訓練模型，此程式為暫時測試使用
    # =============================================================================
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if len(gpus) != 0:  # 若有檢測到GPU，則設定使用第一張GPU，並設定記憶體上限7GB
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
    K.clear_session()

    # 爬蟲取得雷達回波圖、紅外線雲圖
    Y, m, d, H, M = WeatherCrawler().get_format_time()  # 取得時間
    IR, RD = WeatherCrawler().get_weather_images()  # 回傳的IR、RD為bool，告知IR、RD是否存在

    # 範例模型為IR+RD
    RA = False  # True則代表輸入該圖片
    GI = False
    interval = 'EWB01'  # EWB01共100個標籤
    num_classes = 100
    shift = 1  # 預測 t+shift 時刻
    t2t = 't2t+' + str(shift)
    blend = False  # blend參數用於控制是否讓識別結果和原圖混合  False不混合(輸出純白背景)
    choose_model = 'Unet3'  # 模型名稱

    predictor = Predictor(RD=RD, IR=IR, RA=RA, GI=GI, shift=shift, choose_model=choose_model,
                          interval=interval, num_classes=num_classes, blend=blend)
    RD_image = Image.open(f'./img/{Y}{m}{d}{H}{M}_RD_crop.png')  # 匯入雷達圖像
    IR_image = Image.open(f'./img/{Y}{m}{d}{H}{M}_IR_crop.png')  # 匯入IR圖像
    RA_image = 0
    # detect_image_merge_label用於輸出預測出來的陣列
    r_label = predictor.predict(RD_image, IR_image, RA_image)
    csv_filename = f'./img/{Y}{m}{d}{H}{M}_predict.csv'
    predictor.save_prediction(r_label, csv_filename)  # 預測的值儲存成csv
    Plotting().plot_predict(csv_filename)


if __name__ == '__main__':
    main()
