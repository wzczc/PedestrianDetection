import sys
import os
from PyQt5 import uic
from MyUi_test import Ui_TestWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.Qt import QThread
import cv2
import numpy as np

import torch
import argparse
import importlib
from nnet.py_factory import NetworkFactory
from mmcv import Config
import pdb
import time


torch.backends.cudnn.benchmark = False
dirname = '/home/cvlab/anaconda3/envs/csp/lib/python3.8/site-packages/cv2/qt/'
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("--cfg_file", help="config file", default="config/config.py",type=str)
    args = parser.parse_args()
    return args

class DetectThread(QThread):
    imgsignal = pyqtSignal(np.ndarray)
    textsignal = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self.score = 0.5
        self.nms = 0.5
        self.model = 0
        self.img = 0

    def run(self):

        cfg = cfg_file
        cfg.test_cfg.test = True
        cfg.test_cfg.scores_csp = self.score
        cfg.test_cfg.nms_threshold = self.nms
        
        nnet = NetworkFactory(cfg)
        test_file = "test.{}".format(cfg.test_cfg.sample_module)
        testing = importlib.import_module(test_file).testing

        nnet.cuda()
        nnet.eval_mode()

        if self.model:
            nnet.LoadParams(self.model)
        else:
            nnet.load_params(35)

        result_dir = os.path.join(cfg.test_cfg.save_dir, str(35))
        if self.img:
            self.result_img, self.result_text = testing(cfg_file, nnet, self.img, result_dir, debug=False, gui=True)
            self.imgsignal.emit(self.result_img)
            self.textsignal.emit(self.result_text)


class TestUi(QMainWindow,Ui_TestWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.center()
        self.ImgBtn = self.toolButton
        self.ImgBtn.clicked.connect(self.ChooseImg)
        self.DetBtn = self.toolButton_2
        self.DetBtn.clicked.connect(self.Detect)
        # self.VideoBtn = self.toolButton_4
        # self.VideoBtn.clicked.connect(self.ChooseVideo)
        # self.CameraBtn = self.toolButton_5
        # self.CameraBtn.clicked.connect(self.OpenCamera)
        self.ModelBtn = self.toolButton_6
        self.ModelBtn.clicked.connect(self.ChooseModel)
        self.BackBtn = self.toolButton_3
        self.modelsource = [0,0]
        self.source = [0,0]

        self.DetThread = DetectThread()
    
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2),
                  int((screen.height() - size.height()) / 2))


    def ChooseImg(self):
        self.source = QFileDialog.getOpenFileName(self, '选取图片', os.getcwd(), "Pic File(*.jpg *.png)")
        if self.source[0]:
            path = self.source[0].split('/')
            self.label_8.setText(path[-1])
        frame = QImage(self.source[0])
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    # def ChooseVideo(self):
    #     return 
    
    # def OpenCamera(self):
    #     return
    
    def ChooseModel(self):
        self.modelsource = QFileDialog.getOpenFileName(self, '选取模型', os.getcwd(), "Model File(*.tea *.pth)")
        if self.modelsource[0]:
            for i in range(1,len(self.modelsource[0])+1):
                if self.modelsource[0][i:i+5] == 'epoch':
                    self.label_7.setText(self.modelsource[0][i:len(self.modelsource[0])+1])
                    break

    
    def Detect(self): 
        self.DetThread.score = self.slider.value()/100
        self.DetThread.nms = self.slider_2.value()/100
        self.DetThread.model = self.modelsource[0]
        self.DetThread.img = self.source[0]
        self.DetThread.start()
        self.DetThread.imgsignal.connect(self.show_img)
        self.DetThread.textsignal.connect(self.show_text)

    def show_img(self,img):
        from PIL import Image as im
        data = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        frame = im.fromarray(data)
        pix = frame.toqpixmap()
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView_2.setScene(scene)

    def show_text(self,result_text):
        self.textBrowser.clear()
        for texts in result_text:
            resulttext = ""
            for text in texts:
                resulttext = resulttext + text +  " " + "\n"
            self.textBrowser.append(resulttext)
            self.textBrowser.repaint()
        


args = parse_args()
cfg_file = Config.fromfile(args.cfg_file)


if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    Gui = TestUi() 
    
    Gui.show()

    app.exec()