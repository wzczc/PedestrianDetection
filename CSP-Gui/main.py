import sys
import os
from PyQt5 import uic
from traingui import TrainUi
from testgui import TestUi
from track_test_gui import TrackTestUi
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from Ui_MainMenu import Ui_MainWindow
import cv2

import torch
import argparse
import importlib
# from nnet.py_factory import NetworkFactory
# from mmcv import Config
import pdb

dirname = '/home/cvlab/anaconda3/envs/csp/lib/python3.8/site-packages/cv2/qt/'
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class MyUi(QMainWindow,Ui_MainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.center()
        self.train_btn = self.toolButton
        self.test_btn = self.toolButton_5
        self.tracktest_btn = self.toolButton_2
        self.exit_btn = self.toolButton_4

        self.test_gui = TestUi()
        self.train_gui = TrainUi()
        self.tracktest_gui = TrackTestUi()
        # self.track_gui = TrackUi()

        self.test_gui.BackBtn.clicked.connect(self.Back)
        self.train_gui.BackBtn.clicked.connect(self.Back)
        self.tracktest_gui.BackBtn.clicked.connect(self.Back)
        # self.track_gui.BackBtn.clicked.connect(self.Back)

        self.train_btn.clicked.connect(self.train_ui)
        self.tracktest_btn.clicked.connect(self.tracktest_ui)
        self.test_btn.clicked.connect(self.test_ui)
        # self.track_btn.clicked.connect(self.track_ui)
        self.exit_btn.clicked.connect(self.close)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2),
                  int((screen.height() - size.height()) / 2))

    def test_ui(self):
        self.test_gui.show()
        self.hide()
        self.train_gui.hide()
        self.tracktest_gui.hide()

    def train_ui(self):
        self.train_gui.show()
        self.hide()
        self.tracktest_gui.hide()
        self.test_gui.hide()
        # self.track_gui.hide()

    # def track_ui(self):
    #     self.track_gui.show()
    #     self.hide()
    #     self.test_gui.hide()
    #     self.train_gui.hide()
    def tracktest_ui(self):
        self.tracktest_gui.show()
        self.hide()
        self.train_gui.hide() 
        self.test_gui.hide()


    def Back(self):       
        self.show()
        self.test_gui.hide()
        self.train_gui.hide()
        self.tracktest_gui.hide()
        # self.track_gui.hide()


if __name__=="__main__":
    app = QApplication(sys.argv)
    Gui = MyUi()
    Gui.show()
    app.exec()
