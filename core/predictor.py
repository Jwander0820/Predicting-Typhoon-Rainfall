import glob
from utils.metrics import *  # 調用自訂義的評價函數
from utils.model_output import *


class Predictor:
    def __init__(self, RD=False, IR=False, RA=False, GI=False, shift=1, choose_model='Unet3',
                 interval='EWB01', num_classes=100, blend=False):
        self.RD = RD
        self.IR = IR
        self.RA = RA
        self.GI = GI
        self.shift = shift
        self.choose_model = choose_model
        self.interval = interval
        self.num_classes = num_classes
        self.blend = blend
        self.model_path = self._get_last_checkpoint()
        self.model_image_size = self._get_model_image_size()

        self._create_unet_model()  # 呼叫函數建立unet

    def _get_last_checkpoint(self):
        checkpoint = glob.glob('./checkpoint/*.h5')
        for last_checkpoint in checkpoint:  # 取得最近一次的ckpt
            None
        return last_checkpoint

    def _get_model_image_size(self):
        if self.RD == True and self.IR == False and self.RA == False:  # RD
            model_image_size = [128, 128, 3]
        elif self.RD == False and self.IR == False and self.RA == True:  # RA
            model_image_size = [128, 128, 3]
        elif self.RD == False and self.IR == True and self.RA == False:  # IR
            model_image_size = [128, 128, 3]
        elif self.RD == True and self.IR == True and self.RA == False:  # RD+IR
            model_image_size = [128, 128, 4]
        elif self.RD == True and self.IR == False and self.RA == True:  # RD+RA
            model_image_size = [128, 128, 6]
        elif self.RD == False and self.IR == True and self.RA == True:  # IR+RA
            model_image_size = [128, 128, 4]
        elif self.RD == True and self.IR == True and self.RA == True:  # RD+IR+RA
            model_image_size = [128, 128, 7]
        if self.GI:
            model_image_size[2] += 2  # channel+2，GI加在最後一層
        return model_image_size

    def _create_unet_model(self):
        self.unet = Unet(model_path=self.model_path, model_image_size=self.model_image_size,
                         num_classes=self.num_classes, blend=self.blend, RD=self.RD, IR=self.IR,
                         GI=self.GI, RA=self.RA, choose_model=self.choose_model)

    def predict(self, RD_image, IR_image, RA_image):
        r_label = self.unet.detect_image_merge_label(RD_image, IR_image, RA_image)
        return r_label

    def save_prediction(self, r_label, filename):
        np.savetxt(filename, r_label, delimiter=",", fmt='%d')
