# -*- coding: utf-8 -*-
from datetime import datetime,timezone,timedelta
import requests
import urllib.request as req
import os
import time
import cv2
from PIL import Image
dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區
# print('UTC \t%s\nUTC+8\t%s'%(dt1,dt2))
# print(dt2.strftime("%Y-%m-%d %H:%M:%S")) # 將時間轉換為 string

time = dt2.strftime("%Y%m%d%H%M")
Y = dt2.strftime('%Y')
m = dt2.strftime('%m')
d = dt2.strftime('%d')
H = dt2.strftime('%H')
M = dt2.strftime('%M')
M = (int(M)//10) - 2
Mdict = {-2:'40', -1:'50', 0:'00', 1:'10', 2:'20', 3:'30'}
if M < 0: # 跨時
    H = int(H) - 1
    if H < 0: # 跨日
        d = int(d) - 1
        d = str(d)
        H = 23
    H = str(H)
print(time)
print(Y,m,d,H,Mdict[M])
new_Mdict = {"00": "00", "10": "00", "20": "00", "30": "30", "40": "30", "50": "30", }  # 20220521改僅取00和30分的資料
Mdict[M] = new_Mdict[Mdict[M]]
my_headers = {'user-agent': 'my-app/0.0.1'}  #自訂表頭
IR_url = f'https://www.cwb.gov.tw/Data/satellite/TWI_IR1_Gray_800/TWI_IR1_Gray_800-{Y}-{m}-{d}-{H}-{Mdict[M]}.jpg'
RD_url = f'https://www.cwb.gov.tw/Data/radar/CV1_TW_3600_{Y}{m}{d}{H}{Mdict[M]}.png'

try: # 爬取紅外線雲圖至 img 資料夾中
    r = requests.get(IR_url, headers = my_headers,timeout=5) #get請求以及將自訂表頭加入get請求中
    img = r.content       #響應的二進位制檔案
    with open('./img/'+Y+m+d+H+Mdict[M]+'_IR.png','wb') as f:     #二進位制寫入
        f.write(img)
    IR = True
    # 裁切圖像
    img = cv2.imread('./img/'+Y+m+d+H+Mdict[M]+'_IR.png',cv2.IMREAD_GRAYSCALE)
    cropped = img[261:558, 247:544]  # 裁切座標為[y0:y1, x0:x1]
    cv2.imwrite('./img/'+Y+m+d+H+Mdict[M]+'_IR_crop.png', cropped)  
    img = Image.open('./img/'+Y+m+d+H+Mdict[M]+'_IR_crop.png')
    new_img = img.resize((128,128))  #圖像縮放大小
    new_img.save('./img/'+Y+m+d+H+Mdict[M]+'_IR_crop.png')
except:
    IR = False
    print('沒有爬取到紅外線雲圖')
try: # 爬取雷達回波圖至 img 資料夾中
    r1 = requests.get(RD_url, headers = my_headers,timeout=5) #get請求以及將自訂表頭加入get請求中
    img = r1.content       #響應的二進位制檔案
    with open('./img/'+Y+m+d+H+Mdict[M]+'_RD.png','wb') as f:     #二進位制寫入
        f.write(img)
    RD = True
    img = cv2.imread('./img/'+Y+m+d+H+Mdict[M]+'_RD.png',cv2.IMREAD_COLOR)
    cropped = img[716:2762, 830:2876]  # 裁切座標為[y0:y1, x0:x1]
    cv2.imwrite('./img/'+Y+m+d+H+Mdict[M]+'_RD_crop.png', cropped)  
    img = Image.open('./img/'+Y+m+d+H+Mdict[M]+'_RD_crop.png')
    new_img = img.resize((128,128))  #圖像縮放大小
    new_img.save('./img/'+Y+m+d+H+Mdict[M]+'_RD_crop.png')
except:
    print('沒有爬取到雷達回波圖')
    RD = False



    

# =============================================================================
# 接下來將爬取到的資料丟到模型中，產生預測資料
# 預訓練的模型為RD+IR預測未來一小的案例，並放在checkpoint資料夾中
# =============================================================================
#-----------引入基本套件-----------#
import time
import keras
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import os
from keras import backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                              TensorBoard)
from keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from keras.utils.data_utils import get_file
from utils.model_output import *
from PIL import Image   
#-----------引入模型架構-----------#
from nets.generator_training import *  #引入.fir_generator所需要使用的生成器
from nets.unet import *                         #從net/unet.py 中引入Unet模型，內有Unet3、4、5、6
from nets.FCN8 import *                       #FCN8
from nets.pspnet import pspnet           #PSPNET
#-----引入損失函數&評價函數-----#
from utils.loss import *       #調用自訂義的損失函數
from utils.metrics import *  #調用自訂義的評價函數
        
# 範例模型為RD+IR
RA = False  #True則代表輸入該圖片
GI = False
interval = 'EWB01' #EWB01共100個標籤
num_classes = 100
shift = 1 # 預測 t+shift 時刻
t2t = 't2t+'+str(shift)
#--------控制訓練的參數------#
choose_model = 'Unet3'   #模型名稱       

checkpoint = glob.glob('./checkpoint/*.h5') 
for last_checkpoint in checkpoint:  
    None
print(last_checkpoint)

if __name__ == '__main__':
    K.clear_session() 
    blend = False# blend參數用於控制是否讓識別結果和原圖混合  False不混合(輸出純白背景)

    if RD==True and IR==False and RA==False:   #RD
        model_image_size = [128,128,3]
    elif RD==False and IR==False and RA==True:   #RA
        model_image_size = [128,128,3]
    elif RD==False and IR==True and RA==False:  #IR
        model_image_size = [128,128,3]
    elif RD==True and IR==True and RA==False:  #RD+IR
        model_image_size = [128,128,4]
    elif RD==True and IR==False and RA==True:  #RD+RA
        model_image_size = [128,128,6]
    elif RD==False and IR==True and RA==True:  #IR+RA
        model_image_size = [128,128,4]
    elif RD==True and IR==True and RA==True:  #RD+IR+RA
        model_image_size = [128,128,7]
    if GI == True:
        model_image_size[2] = model_image_size[2] +2  #channel+2，GI加在最後一層
    
    model_path = last_checkpoint  #最佳權重
    unet = Unet(model_path=model_path,model_image_size=model_image_size,num_classes=num_classes,
                blend=blend,RD=RD,IR=IR,GI=GI,RA=RA,choose_model=choose_model)
    RD_image = Image.open('./img/'+Y+m+d+H+Mdict[M]+'_RD_crop.png')#匯入雷達圖像
    IR_image = Image.open('./img/'+Y+m+d+H+Mdict[M]+'_IR_crop.png')#匯入IR圖像
    RA_image = 0
    #detect_image_merge_label用於輸出預測出來的陣列
    r_label = unet.detect_image_merge_label(RD_image,IR_image,RA_image)
    np.savetxt('./img/'+Y+m+d+H+Mdict[M]+'_predict.csv', r_label, delimiter=",",fmt='%d',) #預測的值儲存成csv        

nws_precip_colors = [
    "#fdfdfd",  # (253,253,253)   #white 
    "#c1c1c1",  # (193,193,193)  #grays 
    "#99ffff",  # (153,255,255)   #blue 
    "#00ccff",  # (0,204,255)
    "#0099ff",  # (0,153,255)
    "#0166ff",  # (1,102,255)
    "#329900",  # (50,153,0)        #green
    "#33ff00",  # (51,255,0)
    "#ffff00",  # (255,255,0)       #yellow
    "#ffcc00",  # (255,204,0)
    "#fe9900",  # (254,153,0)  
    "#fe0000",  # (254,0,0)          #red
    "#cc0001",  # (204,0,1)
    "#990000",  # (153,0,0)
    "#990099",  # (153,0,153)     #purple
    "#cb00cc",  # (203,0,203)
    "#ff00fe",  # (255,0,254)
    "#feccff"   # (254,204,255)
]
precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)
eplison = 1e-7 
clevels = [0,0.1, 1+eplison, 2+eplison, 6+eplison, 10+eplison, 15+eplison, 20+eplison
            , 30+eplison, 40+eplison, 50+eplison, 70+eplison, 90+eplison,110+eplison
            ,130+eplison,150+eplison, 200+eplison,300+eplison,500+eplison]  #自定義顏色列表
norm = matplotlib.colors.BoundaryNorm(clevels, 18)

predict = np.genfromtxt('./img/'+Y+m+d+H+Mdict[M]+'_predict.csv',delimiter=',') #匯入預測出來的圖像
plt.figure(figsize=(1,1)) #改變圖片尺寸(英吋)   #1英吋對應dpi pixel 
plt.imshow(predict,cmap=precip_colormap, alpha=1, norm=norm)  #alpha為透明度
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  #控制子圖邊框
plt.savefig('./img/'+Y+m+d+H+Mdict[M]+'_predict.png',dpi=128,pad_inches=0.0) 
plt.close()  #關閉圖像

# 注意:因為訓練的資料(颱風資料)和爬蟲下來的一般雷達回波圖和紅外線雲圖有差異，(淺藍底vs白底、不同的框線)
# 所以導致預測的結果為空白的情況，也可能因為模型較淺層較小型所以預測表現自然也不佳
# 若要改善一般預測的情況可能需要重新訓練模型