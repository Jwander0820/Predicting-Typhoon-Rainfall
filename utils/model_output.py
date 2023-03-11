import colorsys
import copy
import os
import cv2

import numpy as np
from PIL import Image

from nets.unet import *
from nets.pspnet import pspnet  #統整pspnet模型
from nets.FCN8 import *
class Unet:  #原始碼採預設形式，已轉為可外部修改的變數
    #---------------------------------------------------#
    def __init__(self,model_path=None,model_image_size=None,num_classes=18,blend=None,
                 RD=None,IR=None,GI=None,RA = None,choose_model=None,
                 backbone = "resnet50",downsample_factor = 8):
        self.model_path =  model_path    #權重的系統路徑
        self.model_image_size = model_image_size  #輸入圖像尺度
        self.num_classes = num_classes  #分類數目
        self.blend = blend  #用於控制是否讓識別结果和原圖混合   False不混合(輸出純白背景)
        self.choose_model = choose_model #選擇何種模型
        self.backbone = "resnet50"
        self.downsample_factor = 8
        self.RA = RA
        self.RD = RD
        self.IR = IR
        self.GI = GI
        self.generate()

    def generate(self):
        #-------------------------------#
        #   載入模型與權重
        #-------------------------------#
        if self.choose_model == 'Unet6':  #判別不同的模型
            self.model = Unet6(self.model_image_size,self.num_classes)
        elif self.choose_model == 'Unet5':
            self.model = Unet5(self.model_image_size,self.num_classes)
        elif self.choose_model == 'Unet4':
            self.model = Unet4(self.model_image_size,self.num_classes)
        elif self.choose_model == 'Unet3':
            self.model = Unet3(self.model_image_size,self.num_classes)
        elif self.choose_model == 'FCN8':
            self.model = FCN8(self.model_image_size,self.num_classes)
        elif self.choose_model == 'pspnet':
            self.model = pspnet(self.model_image_size,self.num_classes,
                    downsample_factor=self.downsample_factor, backbone=self.backbone, aux_branch=False)
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))
        
        if self.num_classes == 2:
            self.colors = [(255, 255, 255),  (0, 0, 0)]
        elif self.num_classes <= 100:  #根據標籤，提供著色的參考
            self.colors = [(253, 253, 253), (193, 193, 193), (153, 255, 255), (0, 204, 255), (0, 153, 255),
                  (1, 102, 255), (50, 153, 0), (51, 255, 0), (255, 255, 0), (255, 204, 0),
                  (254, 153, 0), (254, 0, 0), (204, 0, 1), (153, 0, 0), (153, 0, 153),
                 (203, 0, 203), (255, 0, 254), (254, 204, 255),(0,0,0)]
        else:
            # 畫框設置不同顏色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                        for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))
            
    """本研究使用之融合圖像預測輸出程式"""
    def detect_image_merge_label(self, RD_image,IR_image,RA_image):  
        #用於傳出預測出來的pr陣列!! 
        #因為經度緯度都是同一張圖，所以直接匯入image做疊加即可
        orininal_h = np.array(RD_image).shape[0] #匯入圖像之高
        orininal_w = np.array(RD_image).shape[1] #匯入圖像之寬
        RD_image = np.array(RD_image)/255  #轉為矩陣
        IR_image = np.array(IR_image)/255
        RA_image = np.array(RA_image)
        #RA_image = RA_image[:,:,:-1]
        RA_image = np.array(RA_image)/self.num_classes
        RA_image = cv2.merge([RA_image,RA_image,RA_image])
        # lon = Image.open("./database/train_data_GI/lon.png")  #經度圖像(以0~254 間格2做代替)
        # lat = Image.open("./database/train_data_GI/lat.png")  #緯度圖像(以0~254 間格2做代替)
        # lon = np.array(lon)/255
        # lat = np.array(lat)/255
        
        if self.RD==True and self.IR==False and self.RA==False:   #RD
            res = cv2.merge([RD_image]) #圖像合成
        elif self.RD==False and self.IR==False and self.RA==True:   #RA
            res = cv2.merge([RA_image])
        elif self.RD==False and self.IR==True and self.RA==False:  #IR
            res = cv2.merge([IR_image,IR_image,IR_image]) 
            #若為僅訓練IR的情況，因為(128,128,1)無法訓練，所以轉換成(128,128,3)
        elif self.RD==True and self.IR==True and self.RA==False:  #RD+IR
            res = cv2.merge([RD_image,IR_image])
        elif self.RD==True and self.IR==False and self.RA==True:  #RD+RA
            res = cv2.merge([RD_image,RA_image])
        elif self.RD==False and self.IR==True and self.RA==True:  #IR+RA
            res = cv2.merge([IR_image,RA_image])
        elif self.RD==True and self.IR==True and self.RA==True:  #RD+IR+RA
            res = cv2.merge([RD_image,IR_image,RA_image])
            
        if self.GI == True:  #若GI=True，在最後面添加兩層lon&lat
            res = cv2.merge([res,lon,lat])
            
        merge = np.asarray([np.array(res)]) #assaray的中括號[]很重要，便是將矩陣視為一個向量，組成更大矩陣
        #---------------------------------------------------#
        #   圖片傳入網路進行預測
        #---------------------------------------------------#
        pr = self.model.predict(merge)[0] 
        #若是後面沒有加 [0] pr的型態就會是(1,128,128,100)，[0]代表選用了"索引值=0"的資料出來!!
        #print(np.shape(pr))  #pr是模型輸出的預測，一個(128,128,100)的矩陣，內部值0~1
        #---------------------------------------------------#
        #   取出每一個像素點的種類 #參https://blog.csdn.net/qq1483661204/article/details/78959293
        #   argmax回傳的是沿軸axis最大值的索引值
        #  也就是說在128,128的一點上，垂直去看100層，哪一層的值最高，便會回傳該"索引值"
        #  輸出上也就對應了我們的"標籤值" 輸出為(128,128)的矩陣範圍為0~100 
        #  axis=-1跟=1效果相同
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        
        return pr
    
    """到此為止了┗|｀O′|┛"""
    """以下為原程式的一些相關函數，部分有參考價值作保留，僅供參考"""
    def letterbox_image(self ,image, size):  #不失真的resize
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image,nw,nh

    #---------------------------------------------------#
    #   檢測圖片
    #---------------------------------------------------#
    def detect_image(self, image):  #原始程式碼
        #---------------------------------------------------#
        #   對輸入圖像進行備份，後面用於繪圖
        #---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        #---------------------------------------------------#
        #   進行不失真的resize，添加灰條，進行圖像正規化
        #---------------------------------------------------#
        img, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        img = np.asarray([np.array(img)/255])
        
        #---------------------------------------------------#
        #   圖片傳入網路進行預測
        #---------------------------------------------------#
        pr = self.model.predict(img)[0]
        #---------------------------------------------------#
        #   取出每一個像素點的種類
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        #--------------------------------------#
        #   將灰條部分截取掉
        #--------------------------------------#
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        #------------------------------------------------#
        #   創建一副新圖，並根據每個像素點的種類賦予顏色
        #------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')

        #------------------------------------------------#
        #   將新圖片轉換成Image的形式
        #------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h), Image.NEAREST)

        #------------------------------------------------#
        #  將新圖片與原圖片融合
        #------------------------------------------------#
        if self.blend:
            image = Image.blend(old_img,image,0.7)
        return image

    def detect_image_merge_RDIR(self, image,IR): #無用，僅保存紀錄參考
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        nh=128
        nw=128
        img = np.array(image)
        IR = np.array(IR)
        merge = cv2.merge([img,IR])/255
        merge = np.asarray([np.array(merge)])
        #---------------------------------------------------#
        pr = self.model.predict(merge)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')
        #------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h), Image.NEAREST)
        return image

    def detect_image_merge_IOU(self, RD_image,IR_image,RA_image):  
        #用於傳出預測出來的pr陣列!! 
        #因為經度緯度都是同一張圖，所以直接匯入image做疊加即可
        orininal_h = np.array(RD_image).shape[0] #匯入圖像之高
        orininal_w = np.array(RD_image).shape[1] #匯入圖像之寬
        RD_image = np.array(RD_image)/255  #轉為矩陣
        IR_image = np.array(IR_image)/255
        RA_image = np.array(RA_image)/self.num_classes
        #RA_image = RA_image[:,:,:-1]
        lon = Image.open("./database/train_data_GIS/lon.png")  #經度圖像(以0~254 間格2做代替)
        lat = Image.open("./database/train_data_GIS/lat.png")  #緯度圖像(以0~254 間格2做代替)
        lon = np.array(lon)/255
        lat = np.array(lat)/255
        
        if self.RD==True and self.IR==False and self.GIS==False:   #RD
            res = cv2.merge([RD_image]) #圖像合成
        elif self.RD==False and self.IR==True and self.GIS==False:  #IR
            res = cv2.merge([IR_image,IR_image,IR_image])
        elif self.RD==True and self.IR==True and self.GIS==False:  #RD+IR
            res = cv2.merge([RD_image,IR_image])
        elif self.RD==True and self.IR==False and self.GIS==True:  #RD+GIS
            res = cv2.merge([RD_image,lon,lat])
        elif self.RD==False and self.IR==True and self.GIS==True:  #IR+GIS
            res = cv2.merge([IR_image,lon,lat])
        elif self.RD==True and self.IR==True and self.GIS==True:  #RD+IR+GIS
            res = cv2.merge([RD_image,IR_image,lon,lat])
        if self.rainfallmap == True:
            res = cv2.merge([res,RA_image])
            
        merge = np.asarray([np.array(res)])
        #---------------------------------------------------#
        #   圖片傳入網路進行預測
        #---------------------------------------------------#
        pr = self.model.predict(merge)[0]
        #---------------------------------------------------#
        #   取出每一個像素點的種類
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h), Image.NEAREST)
        return image
