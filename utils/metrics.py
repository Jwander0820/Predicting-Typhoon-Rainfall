import tensorflow as tf
from keras import backend as K
import numpy as np 
from keras.callbacks import Callback
import keras
"""
f-score參考自 https://github.com/BBuf/Keras-Semantic-Segmentation/blob/master/metrics/metrics.py#L11
原Github https://github.com/BBuf/Keras-Semantic-Segmentation 
"""        
SMOOTH = 1.

#threhold閾值就別用了吧..用起來很怪，用了在precision之中會變成0.999，recall fscore不太變且看不出收斂
#關於程式內部具體註解寫在f_score中 
def Iou_score(smooth = SMOOTH, threhold = 0.5):
    def _Iou_score(y_true, y_pred):
         #score calculation
        #y_pred = tf.greater(y_pred, threhold)  
        #y_pred = tf.cast(y_pred, dtype=tf.float32)   #Iou = TP / (FP+TP+FN)
        
        intersection =  tf.reduce_sum(y_true * y_pred, axis=[1,2])  #intersection = TP
        union =  tf.reduce_sum(y_true + y_pred, axis=[1,2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        score = tf.reduce_mean(score ,axis=[0,1])
        return score
    return _Iou_score

    # F_score（Dice系数）可以解釋為精確度和召回率的加權平均值，
    # 其中F-score在1時達到其最佳值，在0時達到最差分数。
    # 精確率和召回率對F1-score的相對影響是一樣的
def f_score(beta=1, smooth = SMOOTH , threhold = 0.5):
    def _f_score(y_true, y_pred): #應該算是macro-F1
        #y_pred = tf.greater(y_pred, threhold) #逐個元素比對y_pred>threhold，返回一個布爾張量
        #大於閾值則=True
        #y_pred = tf.cast(y_pred, dtype=tf.float32)  
        #cast將張量轉換成不同type (此處應該為轉換成floatx()浮點數形式)
        #tf.reduce_sum 後面的axis用於將指定軸降維求和
        tp = tf.reduce_sum(y_true * y_pred, axis=[1,2]) #tf.reduce_sum計算張量在某一指定軸的和
        fp = tf.reduce_sum(y_pred         , axis=[1,2]) - tp  #降維求和出向量(16,128,128,100)=>(100,)
        fn = tf.reduce_sum(y_true, axis=[1,2]) - tp  #也代表了各類別(100類)的TP、FP、FN

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score ,axis=[0,1])  #計算跨維度張量的元素和 
        #在這邊是將前面計算的(100,) 在壓縮求平均 (100,) => x (純量)
        return score
    return _f_score

def precision(smooth = SMOOTH, threhold = 0.5):
    def metric_precision(y_true,y_pred):
        #y_pred = tf.greater(y_pred, threhold)
        #y_pred = tf.cast(y_pred, dtype=tf.float32)
        TP=tf.reduce_sum(y_true * y_pred, axis=[1,2])
        FP=tf.reduce_sum(y_pred         , axis=[1,2]) - TP
        precision=(TP+smooth)/(TP+FP+smooth)
        precision = tf.reduce_mean(precision ,axis=[0,1])
        return precision
    return metric_precision
    
def recall(smooth = SMOOTH, threhold = 0.5):
    def metric_recall(y_true,y_pred):
        #y_pred = tf.greater(y_pred, threhold)
        #y_pred = tf.cast(y_pred, dtype=tf.float32)
        TP= tf.reduce_sum(y_true * y_pred, axis=[1,2])
        FN=tf.reduce_sum(y_true, axis=[1,2]) - TP
        recall=(TP+smooth)/(TP+FN+smooth)
        recall = tf.reduce_mean(recall ,axis=[0,1])
        return recall
    return metric_recall

def mean_RMSE(smooth = SMOOTH, threhold = 0.5): 
    #計算的是預測標籤與真實標籤的RMSE，在EWB01情況下與真實誤差差不多，但是在CWB18、EWB10中標籤沒有映射的話與真實誤差會差很多，映射方法尚未完成...
    #求取整個面(128,128)的RMSE，此方法應該能類推到各種計算指標!多嘗試
    def _RMSE(y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)  
        #y_pred = y_pred * TWmap#限制外圍=0
        #此處tf.argmax功能等同np.argmax將y_pred(173,128,128,100)轉化成(173,128,128) "標籤值"
        y_true = tf.argmax(y_true, axis=-1)
        #計算(128,128)每一格點在哪一層(索引)有最大的機率，另該點=索引 計算173筆資料輸出為(173,128,128)
        RMSE = (tf.reduce_sum((y_pred-y_true)**2,axis=[1,2] ) /4700)**0.5
        #RMSE，張量可以對應元素計算，計算完降維求和(#求的是(128,128)這個面的和,等同np.sum)再除N開根號
        score = tf.reduce_mean(RMSE*1.) #將173筆資料求平均
        return score
    return _RMSE
    
def blank_RMSE(smooth = SMOOTH, threhold = 0.5): 
    #計算真值-空白的RMSE(有限制外圍)，(t+1，173筆資料)其固定值RMSE=5.2895 (mm)
    #因為是標籤相減去計算，所以會跟標籤-真值(浮點數) 略有差異(5.287229170280624)，但整體趨勢完全相同
    def _blank(y_true, y_pred): #注意，計算上要將gen_val的batch size設為1，才會將所有資料都計算過取平均
        #過去使用相同batch size會出現部分資料沒有計算，導致RMSE會不斷小改變
        #是因為ex. 173筆//16 ==10 餘下13筆資料不會算在這一輪RMSE，而是當作下一輪繼續循環輸出
        y_true = tf.argmax(y_true, axis=-1)  
        RMSE = (tf.reduce_sum((y_true)**2,axis=[1,2] ) /4700)**0.5 
        score = tf.reduce_mean(RMSE*1.) #將173筆資料求平均
        return score
    return _blank

        
"""    #K.sum 保存版
SMOOTH = 1e-7
def Iou_score(smooth = SMOOTH, threhold = 0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        #y_pred = K.greater(y_pred, threhold)
        #y_pred = K.cast(y_pred, K.floatx())
        
        intersection = K.sum(y_true * y_pred, axis=[0,1,2]) 
        union = K.sum(y_true + y_pred, axis=[0,1,2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        return score
    return _Iou_score

def f_score(beta=1, smooth = SMOOTH , threhold = 0.5):
    def _f_score(y_true, y_pred):
        #y_pred = K.greater(y_pred, threhold) #逐個元素比對y_pred>threhold，返回一個布爾張量
        #y_pred = K.cast(y_pred, K.floatx())  #cast將張量轉換成不同type (此處應該為轉換成floatx()浮點數形式)

        tp = K.sum(y_true * y_pred, axis=[0,1,2]) #K.sum計算張量在某一指定軸的和
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true, axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        #score = tf.reduce_mean(score * 1.)  #計算跨維度張量的元素和
        return score
    return _f_score

def precision(smooth = SMOOTH, threhold = 0.5):
    def metric_precision(y_true,y_pred):
        #y_pred = K.greater(y_pred, threhold)
        #y_pred = K.cast(y_pred, K.floatx())
        TP=K.sum(y_true * y_pred, axis=[0,1,2])
        FP=K.sum(y_pred         , axis=[0,1,2]) - TP
        precision=(TP+smooth)/(TP+FP+smooth)
        return precision
    return metric_precision
    
def recall(smooth = SMOOTH, threhold = 0.5):
    def metric_recall(y_true,y_pred):
        #y_pred = K.greater(y_pred, threhold)
        #y_pred = K.cast(y_pred, K.floatx())
        TP= K.sum(y_true * y_pred, axis=[0,1,2])
        FN=K.sum(y_true, axis=[0,1,2]) - TP
        recall=(TP+smooth)/(TP+FN+smooth)
        return recall
    return metric_recall
"""  
    
    