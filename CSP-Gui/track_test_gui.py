import sys
import os
from PyQt5 import uic
from Ui_track_test import Ui_TrackTestWindow
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


firstframe_flag = 0
display_flag = 0
det_flag = 0

torch.backends.cudnn.benchmark = False
# plugin_path = '/home/cvlab/anaconda3/envs/csp/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("--cfg_file", help="config file", default="config/config.py",type=str)
    args = parser.parse_args()
    return args

trackerTypes = ['KCF', 'MIL', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
def createTypeTracker(trackerType):
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
    return tracker

class CameraThread(QThread):
    camsignal = pyqtSignal(np.ndarray)
    detsignal = pyqtSignal(np.ndarray)
    capsignal = pyqtSignal(cv2.VideoCapture)
    def __init__(self):
        super().__init__()
        self.cap = 0

    def run(self):
        self.cap = cv2.VideoCapture(0)
        global det_flag,firstframe_flag
        # det_flag = 0
        while self.cap:
            ret,frame = self.cap.read()
            frame = cv2.flip(frame,1)
            if det_flag == 0:
                self.camsignal.emit(frame)
            else:
                firstframe_flag = 1
            if self.cap:
                self.capsignal.emit(self.cap)

class TrackThread(QThread):
    framesignal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.alg = 0
        self.source = -1
        self.results = []
        self.cap = 0

    def run(self):
        tracker_type = self.alg
        # 读取视频
        if self.source != -1:
            if self.source==0 and self.cap:
                ret,firstframe = self.cap.read()
            elif self.source != 0:
                videopath = self.source
                self.cap = cv2.VideoCapture(videopath)
                # 第一帧
                ret, firstframe = self.cap.read()
            # 在第一帧中选取跟踪区域
            boxs = []
            for result in self.results:
                box = []
                box.append(float(result[1]))
                box.append(float(result[2]))
                box.append(float(result[3]))
                box.append(float(result[4]))
                boxs.append(tuple(box))
            print(boxs)
            print(firstframe.shape)
            # 初始化跟踪器
            tracker = cv2.MultiTracker_create()
            for box in boxs:
                if box[0] or box[1] or box[2] or box[3]:
                    tracker.add(createTypeTracker(tracker_type), firstframe, box) 
            print(tracker_type)
            # 按帧读取视频
            while self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if self.source == 0:
                    frame = cv2.flip(frame,1)
                else:
                    frame = frame
                ok, boxs = tracker.update(frame)
                if len(boxs)>0:
                    for box in boxs:
                    # 画出矩形目标区域
                        pt1 = (int(box[0]), int(box[1]))
                        pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2, 1)
                        self.framesignal.emit(frame)
            if self.cap:
                self.cap.release()
        else:
            print('read video/camera error!')


class DetectThread(QThread):
    imgsignal = pyqtSignal(np.ndarray)
    textsignal = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self.score = 0.5
        self.nms = 0.5
        self.model = 0
        self.img = 0
        self.framesize = (480,640)

    def run(self):
        
        cfg = cfg_file
        cfg.test_cfg.test = True
        cfg.test_cfg.scores_csp = self.score
        cfg.test_cfg.nms_threshold = self.nms
        cfg.dataset.size_test = self.framesize
        # print(cfg.dataset.size_test)
        
        nnet = NetworkFactory(cfg)
        test_file = "test.{}".format(cfg.test_cfg.sample_module)
        testing = importlib.import_module(test_file).testing

        nnet.cuda()
        nnet.eval_mode()

        if self.model:
            nnet.LoadParams(self.model)
        else:
            nnet.load_params(35)

        result_dir = os.path.join(cfg.test_cfg.save_dir, 'firstframe')
        global firstframe_flag
        global det_flag
        printflag = 1
        while not firstframe_flag:
            det_flag = 1
            if printflag:
                print('waiting for an image to detect...')
                printflag = 0
        self.result_img, self.result_text = testing(cfg_file, nnet, self.img, result_dir, debug=False, gui=True)
        self.imgsignal.emit(self.result_img)
        self.textsignal.emit(self.result_text)
        # det_flag = 0

class TrackTestUi(QMainWindow,Ui_TrackTestWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.center()
        self.alg = self.comboBox
        self.BackBtn = self.toolButton_7
        self.ModelBtn = self.toolButton_9
        self.VideoBtn = self.toolButton_2
        self.DetBtn = self.toolButton_10
        self.TrackBtn = self.toolButton_8
        self.CameraBtn = self.toolButton_5
        self.KillBtn = self.toolButton_6

        self.ModelBtn.clicked.connect(self.ChooseModel)
        self.VideoBtn.clicked.connect(self.ChooseVideo)
        self.DetBtn.clicked.connect(self.Detect)
        self.TrackBtn.clicked.connect(self.Track)
        self.CameraBtn.clicked.connect(self.OpenCam)
        self.KillBtn.clicked.connect(self.kill)

        self.modelsource = [0,0]
        self.videosource = [0,0]
        self.source = -1
        self.result_txts = []
        self.firstframe = 0
        self.framesize = (480,640)

        self.DetThread = DetectThread()
        self.TrackThread = TrackThread()
        self.CamThread = CameraThread()
        
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2),
                  int((screen.height() - size.height()) / 2))
        
    def ChooseModel(self):
        self.modelsource = QFileDialog.getOpenFileName(self, '选取模型', os.getcwd(), "Model File(*.tea *.pth)")
        if self.modelsource[0]:
            for i in range(1,len(self.modelsource[0])+1):
                if self.modelsource[0][i:i+5] == 'epoch':
                    self.label_6.setText(self.modelsource[0][i:len(self.modelsource[0])+1])
                    break

    def ChooseVideo(self):
        self.videosource = QFileDialog.getOpenFileName(self, '选取视频', os.getcwd(), "Video File(*.mp4)")
        if self.videosource[0]:
            self.source = self.videosource[0]
            global firstframe_flag
            firstframe_flag = 1
            self.framesize = (480,640)
            cap = cv2.VideoCapture(self.videosource[0])
            first,self.firstframe = cap.read()
            self.framesize = self.firstframe.shape[0:2]
            frame = cv2.cvtColor(self.firstframe,cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent= frame.shape
            bytesPerLine = bytesPerComponent* width
            self.image= QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.label_2.setPixmap(QPixmap.fromImage(self.image))
            self.label_2.repaint()
            cap.release()
        else:
            firstframe_flag = 0

    def OpenCam(self):
        global firstframe_flag,det_flag
        firstframe_flag = 0
        det_flag = 0
        self.source = 0
        self.CamThread.start()
        self.CamThread.camsignal.connect(self.show_cam)
        self.CamThread.capsignal.connect(self.send)
    
    def Detect(self):
        self.DetThread.score = self.slider.value()/100
        self.DetThread.nms = self.slider_2.value()/100
        self.DetThread.model = self.modelsource[0]
        self.DetThread.img = self.firstframe
        self.DetThread.framesize = self.framesize
        self.DetThread.start()
        self.label_2.repaint()
        self.DetThread.imgsignal.connect(self.show_img)
        self.DetThread.textsignal.connect(self.accept_results)

    def Track(self):
        global display_flag,det_flag
        display_flag = 1
        # det_flag = 0
        self.TrackThread.alg = self.alg.currentText()
        self.TrackThread.source = self.source
        self.TrackThread.results = self.result_txts
        # self.TrackThread.video = self.video_source[0]
        self.TrackThread.start()
        self.TrackThread.framesignal.connect(self.show_img)

    def show_img(self,img):
        frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= frame.shape
        bytesPerLine = bytesPerComponent* width
        self.image= QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(self.image))
        self.label_2.repaint()
    
    def show_cam(self,img):
        self.firstframe = img
        frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= frame.shape
        bytesPerLine = bytesPerComponent* width
        self.image= QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(self.image))
        self.label_2.repaint()
    
    def accept_results(self,txts):
        self.result_txts = txts
    
    def send(self,cap):
        self.TrackThread.cap = cap

    def kill(self):
        if self.TrackThread.cap:
            self.TrackThread.cap.release()
            self.TrackThread.cap = 0
        if self.CamThread.cap:
            self.CamThread.cap.release()
            self.CamThread.cap = 0


args = parse_args()
cfg_file = Config.fromfile(args.cfg_file)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    Gui = TrackTestUi()
    Gui.show()

    app.exec()
