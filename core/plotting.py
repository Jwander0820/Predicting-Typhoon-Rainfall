import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    def __init__(self):
        self.nws_precip_colors = [
            "#fdfdfd",  # (253,253,253) #white
            "#c1c1c1",  # (193,193,193) #grays
            "#99ffff",  # (153,255,255) #blue
            "#00ccff",  # (0,204,255)
            "#0099ff",  # (0,153,255)
            "#0166ff",  # (1,102,255)
            "#329900",  # (50,153,0)    #green
            "#33ff00",  # (51,255,0)
            "#ffff00",  # (255,255,0)   #yellow
            "#ffcc00",  # (255,204,0)
            "#fe9900",  # (254,153,0)
            "#fe0000",  # (254,0,0)     #red
            "#cc0001",  # (204,0,1)
            "#990000",  # (153,0,0)
            "#990099",  # (153,0,153)   #purple
            "#cb00cc",  # (203,0,203)
            "#ff00fe",  # (255,0,254)
            "#feccff"   # (254,204,255)
        ]
        self.precip_colormap = matplotlib.colors.ListedColormap(self.nws_precip_colors)
        self.epsilon = 1e-7
        self.clevels = [0, 0.1, 1 + self.epsilon, 2 + self.epsilon,
                        6 + self.epsilon, 10 + self.epsilon, 15 + self.epsilon,
                        20 + self.epsilon, 30 + self.epsilon, 40 + self.epsilon,
                        50 + self.epsilon, 70 + self.epsilon, 90 + self.epsilon,
                        110 + self.epsilon, 130 + self.epsilon, 150 + self.epsilon,
                        200 + self.epsilon, 300 + self.epsilon, 500 + self.epsilon]
        self.norm = matplotlib.colors.BoundaryNorm(self.clevels, 18)

    def plot_predict(self, filename):
        predict = np.genfromtxt(filename, delimiter=',')  # 匯入預測出來的圖像(.csv)
        plt.figure(figsize=(1, 1))  # 改變圖片尺寸(英吋)，1英吋對應dpi pixel
        plt.imshow(predict, cmap=self.precip_colormap, alpha=1, norm=self.norm)  # alpha為透明度
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 控制子圖邊框
        plt.savefig(filename.replace('.csv', '.png'), dpi=128, pad_inches=0.0)
        plt.close()  # 關閉圖像
