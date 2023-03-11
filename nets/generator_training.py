from random import shuffle

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image

"""本研究使用之融合圖像生成器"""
class Generator_merge(object):  
    def __init__(self,batch_size,train_lines,image_size,num_classes,RD,IR,GI,RA,interval):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes
        self.RD = RD
        self.IR = IR
        self.GI = GI
        self.RA = RA
        self.interval = interval
        
    def generate(self):
        i = 0  #起始令i=0
        length = len(self.train_lines)  #計算這資料總長度
        inputs = []  #建立輸入清單
        targets = []  #建立目標清單
        while True:  #執行不斷計算的迴圈
            if i == 0:  #第一次使用生成器時將資料打亂
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i] #將第i列資料存至annotation變數中，用於後面提取
            # 從文件中讀取圖像
            name = annotation_line.split(';')[0]  #以 ; 為分隔符號，讀取第0行資料為輸入
            
            if self.RD == True:
                RD = Image.open("./database/train_data_RD/train/"+ name )  #雷達圖像
                RD = np.array(RD)/255
            if self.IR == True:
                IR = Image.open("./database/train_data_IR/train/"+ name )  #紅外線圖像
                IR = np.array(IR)/255
            if self.RA == True:
                RA = Image.open("./database/train_data_RA/"+self.interval+"/train/"+ name )  
                RA = np.array(RA) #雨量分布圖(標籤)
                #RA = RA[:,:,:-1] #除去最後一維透明度...(生成rainfallmap時沒注意變成(128,128,4)了
                RA = np.array(RA)/self.num_classes  #除以標籤數，作正規化
                RA = cv2.merge([RA,RA,RA])
            if self.GI == True:
                lon = Image.open("./database/train_data_GI/lon.png")  #經度圖像(以0~254 間格2做代替)
                lat = Image.open("./database/train_data_GI/lat.png")  #緯度圖像(以0~254 間格2做代替)
                lon = np.array(lon)/255
                lat = np.array(lat)/255
            
                
            if self.RD==True and self.IR==False and self.RA==False:   #RD
                res = cv2.merge([RD]) #圖像合成
            elif self.RD==False and self.IR==False and self.RA==True:   #RA
                res = cv2.merge([RA])
            elif self.RD==False and self.IR==True and self.RA==False:  #IR
                res = cv2.merge([IR,IR,IR])   #(應該可以啦但是要改很多東西，就懶)
                #若為僅訓練IR的情況，因為(128,128,1)無法訓練，所以轉換成(128,128,3)
            elif self.RD==True and self.IR==True and self.RA==False:  #RD+IR
                res = cv2.merge([RD,IR])
            elif self.RD==True and self.IR==False and self.RA==True:  #RD+RA
                res = cv2.merge([RD,RA])
            elif self.RD==False and self.IR==True and self.RA==True:  #IR+RA
                res = cv2.merge([IR,RA])
            elif self.RD==True and self.IR==True and self.RA==True:  #RD+IR+RA
                res = cv2.merge([RD,IR,RA])
                
            if self.GI == True:  #若GI=True，在最後面添加兩層lon&lat
                res = cv2.merge([res,lon,lat])
                
            inputs.append(np.array(res))  #將合成完的矩陣除上255(標準化)後加入input清單之中
            
            name = (annotation_line.split(';')[1]).replace("\n", "")#因為讀入資料為'xxx';'xxx\n'所以要將\n替換成空值
            png = Image.open("./database/train_data_RA/"+self.interval+"/train/"+ name )
            #從train_data_RA調用資料，根據清單去調用訓練用的資料
            png = np.array(png)  #轉為陣列 
            png[png >= self.num_classes] = self.num_classes-1  #若是有大於num_class之值便歸類為最大值-1
            #!!注意所以有上面這一行，也可以計算num_class=12之類的情況!且資料集無須更動!! (-1)避免索引溢位
            # 轉化成one_hot的形式  #np.eye為產生斜對角為1的矩陣  #reshape([-1])為將陣列轉換為一維資料
            seg_labels = np.eye(self.num_classes)[png.reshape([-1])]   
            #將128*128=16384筆資料轉為(16384,19) 
            seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes))
            #再將one-hot資料(16384,19) 重塑形回(128,128,num_class)的矩陣，用作於預測的目標
            targets.append(seg_labels) #將轉化完的資料加入target清單之中
            i = (i + 1) % length  # % 為整數除法，並取餘數，此處應該適用於資料增強時判斷要選用第幾筆資料
            #單純一點看就是while迴圈裡 i = i + 1 ，變數i的計數器用於判斷這是匯入第幾筆的資料了
            if len(targets) == self.batch_size:  #如果target長度==batch size大小時便輸出(yield)一組資料(一個批次)
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []  #輸出完資料後，重新建立一個清單(用於清空舊的資料)
                targets = []
                yield tmp_inp, tmp_targets  #傳出這一batch的資料，如此算one step
                
class Generator_merge_val(object):  
    def __init__(self,batch_size,train_lines,image_size,num_classes,RD,IR,GI,RA,interval):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes
        self.RD = RD
        self.IR = IR
        self.GI = GI
        self.RA = RA
        self.interval = interval
        
    def generate(self):
        i = 0  #起始令i=0
        length = len(self.train_lines)  #計算這資料總長度
        inputs = []  #建立輸入清單
        targets = []  #建立目標清單
        while True:  #執行不斷計算的迴圈
            if i == 0:  #第一次使用生成器時將資料打亂
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i] #將第i列資料存至annotation變數中，用於後面提取
            # 從文件中讀取圖像
            name = annotation_line.split(';')[0]  #以 ; 為分隔符號，讀取第0行資料為輸入
            
            if self.RD == True:
                RD = Image.open("./database/train_data_RD/val/"+ name )  #雷達圖像
                RD = np.array(RD)/255
            if self.IR == True:
                IR = Image.open("./database/train_data_IR/val/"+ name )  #紅外線圖像
                IR = np.array(IR)/255
            if self.RA == True:
                RA = Image.open("./database/train_data_RA/"+self.interval+"/val/"+ name )  
                RA = np.array(RA) #雨量分布圖(標籤)
                #RA = RA[:,:,:-1] #除去最後一維透明度...(生成rainfallmap時沒注意變成(128,128,4)了
                RA = np.array(RA)/self.num_classes  #除以標籤數，作正規化
                RA = cv2.merge([RA,RA,RA])
            if self.GI == True:
                lon = Image.open("./database/train_data_GI/lon.png")  #經度圖像(以0~254 間格2做代替)
                lat = Image.open("./database/train_data_GI/lat.png")  #緯度圖像(以0~254 間格2做代替)
                lon = np.array(lon)/255
                lat = np.array(lat)/255
            
                
            if self.RD==True and self.IR==False and self.RA==False:   #RD
                res = cv2.merge([RD]) #圖像合成
            elif self.RD==False and self.IR==False and self.RA==True:   #RA
                res = cv2.merge([RA])
            elif self.RD==False and self.IR==True and self.RA==False:  #IR
                res = cv2.merge([IR,IR,IR])   #(應該可以啦但是要改很多東西，就懶)
                #若為僅訓練IR的情況，因為(128,128,1)無法訓練，所以轉換成(128,128,3)
            elif self.RD==True and self.IR==True and self.RA==False:  #RD+IR
                res = cv2.merge([RD,IR])
            elif self.RD==True and self.IR==False and self.RA==True:  #RD+RA
                res = cv2.merge([RD,RA])
            elif self.RD==False and self.IR==True and self.RA==True:  #IR+RA
                res = cv2.merge([IR,RA])
            elif self.RD==True and self.IR==True and self.RA==True:  #RD+IR+RA
                res = cv2.merge([RD,IR,RA])
                
            if self.GI == True:  #若GI=True，在最後面添加兩層lon&lat
                res = cv2.merge([res,lon,lat])
                
            inputs.append(np.array(res))  #將合成完的矩陣除上255(標準化)後加入input清單之中
            
            name = (annotation_line.split(';')[1]).replace("\n", "")#因為讀入資料為'xxx';'xxx\n'所以要將\n替換成空值
            png = Image.open("./database/train_data_RA/"+self.interval+"/val/"+ name )
            #從train_data_RA調用資料，根據清單去調用訓練用的資料
            png = np.array(png)  #轉為陣列 
            png[png >= self.num_classes] = self.num_classes-1  #若是有大於num_class之值便歸類為最大值-1
            #!!注意所以有上面這一行，也可以計算num_class=12之類的情況!且資料集無須更動!! (-1)避免索引溢位
            # 轉化成one_hot的形式  #np.eye為產生斜對角為1的矩陣  #reshape([-1])為將陣列轉換為一維資料
            seg_labels = np.eye(self.num_classes)[png.reshape([-1])]   
            #將128*128=16384筆資料轉為(16384,19) 
            seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes))
            #再將one-hot資料(16384,19) 重塑形回(128,128,num_class)的矩陣，用作於預測的目標
            targets.append(seg_labels) #將轉化完的資料加入target清單之中
            i = (i + 1) % length  # % 為整數除法，並取餘數，此處應該適用於資料增強時判斷要選用第幾筆資料
            #單純一點看就是while迴圈裡 i = i + 1 ，變數i的計數器用於判斷這是匯入第幾筆的資料了
            if len(targets) == self.batch_size:  #如果target長度==batch size大小時便輸出(yield)一組資料(一個批次)
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []  #輸出完資料後，重新建立一個清單(用於清空舊的資料)
                targets = []
                yield tmp_inp, tmp_targets  #傳出這一batch的資料，如此算one step
"""到此為止了┗|｀O′|┛"""

"""以下為原始的程式碼；功能：.fit_generator的生成器"""
def rand(a=0, b=1): #亂數函式
    return np.random.rand()*(b-a) + a

def letterbox_image(image, label , size):  #不失真的resize
    label = Image.fromarray(np.array(label))

    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size #image大小 ex(128,128)
    w, h = size  #令個別變數w,h = 128(對應的長寬，後續用於縮放)
    scale = min(w/iw, h/ih)  #比例，取最小的(=最短邊)若長寬相等則scale=1
    nw = int(iw*scale)  #縮放比例，原始長寬*scale (若長寬相等此處不影響)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC) #resize成最小的情況
    new_image = Image.new('RGB', size, (128,128,128)) 
    #生成一張128*128大小的彩色(三通道矩陣)並賦值(128,128,128)
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))  
    #將剛剛生成的圖象作為底圖，並貼上image(覆蓋上去)，後面的(w-nw...)用於計算從"哪裡"開始貼上
    #以左上角為0，若=(10,30)則從第10列30行貼上image，在長寬相同的情況下w-nw=0 
    #代表從(0,0)開始貼上image 也就是等於 圖片直接貼上，此處主要用於長寬不同的處理方式
    #雙斜線代表地板除(floor)，先做除法在向下取整數，最後輸出會是int的形式
    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0)) 
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label


class Generator(object):  #原始程式碼
    def __init__(self,batch_size,train_lines,image_size,num_classes):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))  #用於圖像增強，隨機處理

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label = label.convert("L")
        
        # flip image or not
        flip = rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image,label
        
    def generate(self, random_data = False): #此處更改為False因為不需要增強 
        i = 0  
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:  #第一次使用生成器時將資料打亂
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            
            #name = annotation_line.split()[0]
            # 從文件中讀取圖像
            name = annotation_line.split(';')[0]
            jpg = Image.open("./dataset2/train/"+ name )
            name = (annotation_line.split(';')[1]).replace("\n", "")#因為讀入資料為'xxx';'xxx\n'所以要將\n替換成空值
            png = Image.open("./dataset2/trainannot/"+ name )

            if random_data:  #如果想要資料隨機增強，便將random_lines設為True
                jpg, png = self.get_random_data(jpg,png,(int(self.image_size[1]),int(self.image_size[0])))
            else:  #好像此處會讓即使是IR(128,128,1)也會轉為三維陣列
                jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]),int(self.image_size[0])))
            
            png = Image.fromarray(np.array(png))  #Image.fromarray將陣列值再轉化為Image圖像
            
            inputs.append(np.array(jpg)/255)
            
            # 從文件中讀取圖像
            png = np.array(png)
            png[png >= self.num_classes] = self.num_classes
            
            # 轉化成one_hot的形式  
            #num_class再+1的操作不確定實際用處，刪除+調整損失函數一樣能運行，此處保留原始碼，若要運行原始的生成器需要調整上方損失函數，將y_true 改為 y_true[...,:-1]  #推測為除去最後一維?
            seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
            seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes+1))
            
            targets.append(seg_labels)
            i = (i + 1) % length
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_targets
"""以上為原始的程式碼"""