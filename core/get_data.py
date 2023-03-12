import os
from datetime import datetime, timedelta

import cv2
import requests


class WeatherCrawler:
    def __init__(self):
        self.my_headers = {'user-agent': 'my-app/0.0.1'}

    def get_format_time(self, specified_time=None):
        if specified_time:
            dt = specified_time
        else:
            dt = datetime.now()

        if 0 <= dt.minute < 30:
            # 若時間介於00~30之間，回推至一小時前的30分
            dt = dt.replace(minute=30, second=0, microsecond=0)
            dt = dt - timedelta(hours=1)
        else:
            # 若時間介於30~60之間，回推至當前小時的00分
            dt = dt.replace(minute=0, second=0, microsecond=0)

        Y, m, d = str(dt.year).zfill(2), str(dt.month).zfill(2), str(dt.day).zfill(2)
        H = str(dt.hour).zfill(2)
        M = str(dt.minute).zfill(2)
        return Y, m, d, H, M

    def _get_image(self, url, filename, crop=None):
        try:
            # 取得圖片並儲存
            r = requests.get(url, headers=self.my_headers, timeout=5)
            img = r.content
            with open(filename, 'wb') as f:
                f.write(img)
            # 裁切圖像
            if crop:
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                cropped = img[261:558, 247:544]
            else:
                img = cv2.imread(filename, cv2.IMREAD_COLOR)
                cropped = img[716:2762, 830:2876]
            # 縮放圖像
            resized = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
            # 儲存裁切圖像
            root, ext = os.path.splitext(filename)
            img_path = f'{root}_crop.png'
            cv2.imwrite(img_path, resized)
            return True
        except Exception as e:
            print(f"沒有爬取到{filename}")
            print(e)
            return False

    def _get_weather_images(self, specified_time=None):
        """根據輸入的年月日時分，獲取對應的紅外線雲圖和雷達回波圖。"""
        Y, m, d, H, M = self.get_format_time(specified_time=specified_time)
        print(f"data time {Y, m, d, H, M}")
        IR_url = f'https://www.cwb.gov.tw/Data/satellite/TWI_IR1_Gray_800/TWI_IR1_Gray_800-{Y}-{m}-{d}-{H}-{M}.jpg'
        RD_url = f'https://www.cwb.gov.tw/Data/radar/CV1_TW_3600_{Y}{m}{d}{H}{M}.png'

        if not os.path.exists('./img'):
            os.makedirs('./img')
        IR = self._get_image(IR_url, f'./img/{Y}{m}{d}{H}{M}_IR.png', crop=True)
        RD = self._get_image(RD_url, f'./img/{Y}{m}{d}{H}{M}_RD.png', crop=False)
        return IR, RD

    def get_weather_images(self, specified_time=None):
        """獲取當前本地時間對應的紅外線雲圖和雷達回波圖"""
        IR, RD = self._get_weather_images(specified_time)
        return IR, RD


if __name__ == '__main__':
    wc = WeatherCrawler()
    IR, RD = wc.get_weather_images()

    # 模擬datatime測試指定時間格式
    # dt = datetime(2023, 3, 12, 00, 20)  # 應轉換為 ('2023', '3', '11', '23', '30')
    # dt = datetime(2023, 3, 12, 00, 50)  # 應轉換為 ('2023', '3', '12', '0', '00')
    # print(wc.get_format_time(dt))
