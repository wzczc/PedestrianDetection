import sys
import os
from PyQt5 import uic
from Ui_track import Ui_TrackWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import cv2
import numpy as np
from PIL import Image as im


dirname = '/home/cvlab/anaconda3/envs/csp/lib/python3.8/site-packages/cv2/qt/'
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

camera_flag = 0
video_flag = 0

# 创建一个跟踪器，algorithm: KCF、CSRT、DaSiamRPN、GOTURM、MIL
trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
# 创建一个跟踪器，algorithm: KCF、CSRT、DaSiamRPN、GOTURM、MIL
tracker_types = ['MIL', 'KCF', 'CSRT', 'DaSiamRPN', 'GOTURM']
def createTypeTracker(type):
    if type == tracker_types[0]:
        tracker = cv2.TrackerMIL_create()
    elif type == tracker_types[1]:
        tracker = cv2.TrackerKCF_create()
    elif type == tracker_types[2]:
        tracker = cv2.TrackerCSRT_create()
    elif type == tracker_types[3]:
        tracker = cv2.TrackerDaSiamRPN_create()
    elif type == tracker_types[4]:
        tracker = cv2.TrackerGOTURN_create()
    else:
        tracker = None
    return tracker


class CameraThread(QThread):
    camsignal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.alg = 0

    def run(self):
        tracker_type = self.alg
        print(tracker_type)
        tracker = createTypeTracker(tracker_type)
        # 读取视频
        self.cap = cv2.VideoCapture(0)
        global camera_flag
        camera_flag = 1
        # 第一帧
        ret, firstFrame = self.cap.read()
        firstFrame = cv2.flip(firstFrame,1)
        self.camsignal.emit(firstFrame)
        # 在第一帧中选取跟踪区域
        box = cv2.selectROI('select ROI @1st Frame', firstFrame)

        # 初始化跟踪器
        ok = tracker.init(firstFrame, box)
        cv2.destroyWindow('select ROI @1st Frame')
        # 按帧读取视频
        while self.cap:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame,1)
            if not ret:
                print('read video error!')
                break
            # 计时器
            timer = cv2.getTickCount()
            ok, box = tracker.update(frame)
            # print(box)
            # box=(x,y,h,w) 为一个四元素元组，前两个为矩形的左上角顶点坐标，后两个为矩形的尺寸
            # 计算帧率
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)     
            if ok:
                # 画出矩形目标区域
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2, 1)
                self.camsignal.emit(frame)
                # cv2.imshow('track',frame)
                if cv2.waitKey(1)==ord('q'):
                    break
            else:
                # 显示跟踪失败
                cv2.putText(frame, 'track faild!')

class TrackThread(QThread):
    videosignal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.alg = 0
        self.video = 0

    def run(self):
        tracker_type = self.alg
        print(tracker_type)
        tracker = createTypeTracker(tracker_type)
        # 读取视频
        if self.video:
            videopath = self.video
            self.cap = cv2.VideoCapture(videopath)
            global video_flag
            video_flag = 1
            # 第一帧
            ret, firstFrame = self.cap.read()
            # 在第一帧中选取跟踪区域
            box = cv2.selectROI('select ROI @1st Frame', firstFrame)

            # 初始化跟踪器
            ok = tracker.init(firstFrame, box)
            cv2.destroyWindow('select ROI @1st Frame')
            # 按帧读取视频
            while self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    print('read video error!')
                    break
                # 计时器
                timer = cv2.getTickCount()
                ok, box = tracker.update(frame)
                # print(box)
                # box=(x,y,h,w) 为一个四元素元组，前两个为矩形的左上角顶点坐标，后两个为矩形的尺寸
                # 计算帧率
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)     
                if ok:
                    # 画出矩形目标区域
                    pt1 = (int(box[0]), int(box[1]))
                    pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2, 1)
                    self.videosignal.emit(frame)
                    # cv2.imshow('track',frame)
                    if cv2.waitKey(1)==ord('q'):
                        break
                else:
                    # 显示跟踪失败
                    cv2.putText(frame, 'track faild!')
        else:
            print('read video error!')
        
        
class TrackUi(QMainWindow,Ui_TrackWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.alg = self.comboBox
        self.VideoTrackBtn = self.toolButton_2
        self.VideoTrackBtn.clicked.connect(self.VideoTrack)
        self.VideoBtn = self.toolButton_4
        self.VideoBtn.clicked.connect(self.ChooseVideo)
        self.CameraBtn = self.toolButton_5
        self.CameraBtn.clicked.connect(self.CameraTrack)
        self.KillBtn = self.toolButton_6
        self.KillBtn.clicked.connect(self.Kill)
        self.BackBtn = self.toolButton_7

        # self.TrackThread = TrackThread()
        # self.CameraThread = CameraThread()

        self.video_source = [0,0]

    def ChooseVideo(self):
        self.video_source = QFileDialog.getOpenFileName(self, '选取视频', os.getcwd(), "Video File(*.mp4)")
        if self.video_source[0]:
            self.label.setText(self.video_source[0])
        

    def CameraTrack(self):
        self.CameraThread = CameraThread()
        self.CameraThread.alg = self.alg.currentText()
        self.CameraThread.start()
        self.CameraThread.camsignal.connect(self.show_img)
    
    def Kill(self):
        global video_flag
        global camera_flag
        if video_flag:
            self.TrackThread.cap.release()
            video_flag = 0
        if camera_flag:
            self.CameraThread.cap.release()
            camera_flag = 0

    def VideoTrack(self):
        self.TrackThread = TrackThread()
        self.TrackThread.alg = self.alg.currentText()
        self.TrackThread.video = self.video_source[0]
        self.TrackThread.start()
        self.TrackThread.videosignal.connect(self.show_img)

    def show_img(self,img):
        frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= frame.shape
        bytesPerLine = bytesPerComponent* width
        self.image= QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(self.image))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    Gui = TrackUi()
    Gui.show()

    app.exec()


