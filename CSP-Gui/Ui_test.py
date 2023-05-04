# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\PedestrianDetectionCodes\PyQt\CSP_Det\test.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class MySlider(QSlider):  # 继承QSlider
    customSliderClicked = pyqtSignal(str)  # 创建信号

    def __init__(self, parent=None):
        super(QSlider, self).__init__(parent)

    def mousePressEvent(self, QMouseEvent):  # 重写的鼠标点击事件
        super().mousePressEvent(QMouseEvent)
        pos = QMouseEvent.pos().x() / self.width()
        self.setValue(round(pos * (self.maximum() - self.minimum()) + self.minimum()))  # 设定滑动条滑块位置为鼠标点击处
        self.customSliderClicked.emit("mouse Press")  # 发送信号


class Ui_TestWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1680, 677)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(10, 10, 151, 61))
        self.toolButton.setObjectName("toolButton")
        self.toolButton_2 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_2.setGeometry(QtCore.QRect(170, 10, 141, 61))
        self.toolButton_2.setObjectName("toolButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(290, 110, 121, 41))
        font = QtGui.QFont()
        font.setFamily("a_PlakatTitulCm")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setLineWidth(1)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setIndent(-1)
        self.label.setOpenExternalLinks(False)
        self.label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(970, 110, 121, 41))
        font = QtGui.QFont()
        font.setFamily("a_PlakatTitulCm")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setLineWidth(1)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setScaledContents(False)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setWordWrap(True)
        self.label_2.setIndent(-1)
        self.label_2.setOpenExternalLinks(False)
        self.label_2.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1470, 110, 121, 41))
        font = QtGui.QFont()
        font.setFamily("a_PlakatTitulCm")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setLineWidth(1)
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setScaledContents(False)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setWordWrap(True)
        self.label_3.setIndent(-1)
        self.label_3.setOpenExternalLinks(False)
        self.label_3.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label_3.setObjectName("label_3")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(40, 160, 640, 480))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(720, 160, 640, 480))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(1400, 160, 261, 480))
        self.textBrowser.setObjectName("textBrowser")
        self.toolButton_3 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_3.setGeometry(QtCore.QRect(1550, 10, 101, 51))
        self.toolButton_3.setObjectName("toolButton_3")

        self.slider = MySlider(self.centralwidget)#水平方向
        self.slider.setGeometry(QtCore.QRect(540, 20, 340, 50))
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)#设置最小值
        self.slider.setMaximum(100)#设置最大值
        self.slider.setSingleStep(1)#设置步长值
        self.slider.setValue(50)#设置当前值
        self.slider.setTickPosition(QSlider.TicksRight)#设置刻度位置，在下方
        self.slider.setTickInterval(10)#设置刻度间隔
        self.slider.valueChanged.connect(self.changeVal)

        self.slider_2 = MySlider(self.centralwidget)#水平方向
        self.slider_2.setGeometry(QtCore.QRect(1080, 20, 340, 50))
        self.slider_2.setOrientation(QtCore.Qt.Horizontal)
        self.slider_2.setMinimum(0)#设置最小值
        self.slider_2.setMaximum(100)#设置最大值
        self.slider_2.setSingleStep(1)#设置步长值
        self.slider_2.setValue(50)#设置当前值
        self.slider_2.setTickPosition(QSlider.TicksRight)#设置刻度位置，在下方
        self.slider_2.setTickInterval(10)#设置刻度间隔
        self.slider_2.valueChanged.connect(self.changeVal)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(380, 30, 150, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(920, 30, 150, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1680, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.toolButton.setText(_translate("MainWindow", "选择图片"))
        self.toolButton_2.setText(_translate("MainWindow", "开始检测"))
        self.label.setText(_translate("MainWindow", "输入"))
        self.label_2.setText(_translate("MainWindow", "输出"))
        self.label_3.setText(_translate("MainWindow", "检测结果"))
        self.toolButton_3.setText(_translate("MainWindow", "返回主菜单"))
        self.label_4.setText(_translate("MainWindow", "得分阈值：50"))
        self.label_5.setText(_translate("MainWindow", "NMS阈值：50"))

    def changeVal(self, value):
        sender = self.sender()
        if sender == self.slider:
            self.slider.setValue(value)
            self.label_4.setText('得分阈值：' + str(value))
        elif sender == self.slider_2:
            self.slider_2.setValue(value)
            self.label_5.setText('NMS阈值：' + str(value))
        