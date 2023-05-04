import sys
import os
from PyQt5 import uic
from testgui import TestUi
from traingui import TrainUi
from trackgui import TrackUi
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from Ui_MainMenu import Ui_MainWindow
import cv2

import torch
import argparse
import importlib
from nnet.py_factory import NetworkFactory
from mmcv import Config
import pdb


class MyUi(QMainWindow,Ui_MainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.train_btn = self.toolButton
        self.test_btn = self.toolButton_2
        self.track_btn = self.toolButton_4
        self.exit_btn = self.toolButton_3

        self.test_gui = TestUi()
        self.train_gui = TrainUi()
        self.track_gui = TrackUi()

        self.test_gui.BackBtn.clicked.connect(self.Back)
        self.train_gui.BackBtn.clicked.connect(self.Back)
        
        self.train_btn.clicked.connect(self.train_ui)
        self.test_btn.clicked.connect(self.test_ui)
        self.track_btn.clicked.connect(self.track_ui)
        self.exit_btn.clicked.connect(self.close)

    def train_ui(self):
        self.train_gui.show()
        self.hide()
        self.test_gui.hide()
        self.track_gui.hide()

    def test_ui(self):
        self.test_gui.show()
        self.hide()
        self.train_gui.hide()
        self.track_gui.hide()

    def track_ui(self):
        self.track_gui.show()
        self.hide()
        self.train_gui.hide()
        self.test_gui.hide()
    
    def Back(self):       
        self.show()
        self.test_gui.hide()
        self.train_gui.hide()
        self.track_gui.hide()


if __name__=="__main__":
    app = QApplication(sys.argv)
    Gui = MyUi()
    Gui.show()
    app.exec()
