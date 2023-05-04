import sys
import os
from PyQt5 import uic
from Ui_train import Ui_TrainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.Qt import QThread
import cv2

import torch
import numpy as np
import argparse
import importlib
import traceback
import pickle
import json
import time

from mmcv import Config
from mmcv.utils import get_logger
from nnet import NetworkFactory
from torch.multiprocessing import Process, Queue
import pdb

import random
from train import prefetch_data, init_parallel_jobs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deteministic = True
def parse_args():
    parser = argparse.ArgumentParser(description="Train CSP")
    parser.add_argument("--config", help="train config file path", default="config/config.py", type=str)
    args = parser.parse_args()
    return args

args = parse_args()
cfg_file = Config.fromfile(args.config)


class TrainThread(QThread):
    logsignal = pyqtSignal(str)
    configsignal = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.batchsize = 16
        self.batch_per_gpu = [8,8]
        self.epochs = 100
        self.imgs_per_epoch = 2000
        self.learning_rate = 0.0001
        self.display = 20
        self.center_weight = 0.01
        self.scale_weight = 1
        self.offset_weight = 0.1
        self.start_epoch = 0
        self.seed = 7

    def run(self):
        if not os.path.exists(cfg_file.train_cfg.work_dir):
            os.makedirs(cfg_file.train_cfg.work_dir)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(cfg_file.train_cfg.work_dir, f'{timestamp}.log')
        logger = get_logger(name='CAP', log_file=log_file)
        json_file = os.path.join(cfg_file.train_cfg.work_dir, f'{timestamp}.json')

        cfg_file.dataset.batch_size = self.batchsize
        cfg_file.train_cfg.chunk_sizes = self.batch_per_gpu
        cfg_file.train_cfg.num_epochs = self.epochs
        cfg_file.train_cfg.iter_per_epoch = self.imgs_per_epoch
        cfg_file.train_cfg.learning_rate = self.learning_rate
        cfg_file.train_cfg.display = self.display
        cfg_file.kp_head.csp_center_loss.loss_weight = self.center_weight
        cfg_file.kp_head.regr_h_loss.loss_weight = self.scale_weight
        cfg_file.kp_head.regr_offset_loss.loss_weight = self.offset_weight
        start_epoch = self.start_epoch
        seed = self.seed

        logger.info("system config...")
        logger.info(f'Config:\n{cfg_file.pretty_text}')
        self.configsignal.emit(f'Config:\n{cfg_file.pretty_text}')
        # self.textBrowser.append("system config...")
        # self.textBrowser.append(f'Config:\n{cfg_file.pretty_text}')        

        set_seed(int(seed))

        learning_rate    = cfg_file.train_cfg.learning_rate
        pretrained_model = cfg_file.train_cfg.pretrain
        display          = cfg_file.train_cfg.display
        sample_module    = cfg_file.train_cfg.sample_module
        iter_per_epoch   = cfg_file.train_cfg.iter_per_epoch
        num_epochs       = cfg_file.train_cfg.num_epochs
        batch_size       = cfg_file.dataset.batch_size

        # queues storing data for training
        training_queue   = Queue(cfg_file.train_cfg.prefetch_size)

        # load data sampling function
        data_file   = "sample.{}".format(sample_module)
        sample_data = importlib.import_module(data_file).sample_data

        if cfg_file.train_cfg.cache_ped:
            with open(cfg_file.train_cfg.cache_ped, 'rb') as fid:
                ped_data = pickle.load(fid)
        if cfg_file.train_cfg.cache_emp:
            with open(cfg_file.train_cfg.cache_emp, 'rb') as fid:
                emp_data = pickle.load(fid)
        length_dataset = len(ped_data)+len(emp_data)
        logger.info('the length of dataset is: {}'.format(length_dataset))
        # self.textBrowser.append('the length of dataset is: {}'.format(length_dataset))

        # allocating resources for parallel reading
        if cfg_file.train_cfg.cache_emp:
            training_tasks   = init_parallel_jobs(cfg_file, training_queue, sample_data, ped_data, emp_data)
        else:
            training_tasks = init_parallel_jobs(cfg_file, training_queue, sample_data, ped_data)
        # prefetch_data(cfg, training_queue, sample_data, ped_data, emp_data)

        logger.info("building model...")
        # self.textBrowser.append("building model...")

        nnet = NetworkFactory(cfg_file)

        if pretrained_model is not None:
            if not os.path.exists(pretrained_model):
                raise ValueError("pretrained model does not exist")
            logger.info("loading from pretrained model")
            nnet.load_pretrained_params(pretrained_model)

        if start_epoch:
            nnet.load_params(start_epoch)
            nnet.set_lr(learning_rate)
            logger.info("training starts from iteration {} with learning_rate {}".format(start_epoch, learning_rate))
            # self.textBrowser.append("training starts from iteration {} with learning_rate {}".format(start_epoch, learning_rate))
        else:
            nnet.set_lr(learning_rate)

        logger.info("training start...")
        # self.textBrowser.append("training start...")
        # self.textBrowser.repaint()
        nnet.cuda()
        nnet.train_mode()
        epoch_length = int(iter_per_epoch / batch_size)
        json_obj = open(json_file, 'w')
        loss = []
        for epoch in range(start_epoch, num_epochs):
            for iteration in range(1, epoch_length + 1):
                training = training_queue.get(block=True)
                training_loss = nnet.train(**training)

                loss.append(training_loss.item())

                if display and iteration % display == 0:
                    loss = np.array(loss)
                    logger.info("Epoch: {}/{}, loss_csp: {:.5f}".format(epoch+1, num_epochs, loss.sum() / display))
                    self.logsignal.emit("Epoch: {}/{}, loss_csp: {:.5f}".format(epoch+1, num_epochs, loss.sum() / display))
                    # self.textBrowser.append("Epoch: {}/{}, loss_csp: {:.5f}".format(epoch+1, num_epochs, loss.sum() / display))
                    # self.textBrowser.repaint()

                    text = {"Epoch": epoch+1, "loss_csp": round(loss.sum() / display, 5)}
                    text = json.dumps(text)
                    json_obj.write(text)
                    json_obj.write('\r\n')
                    loss = []

                del training_loss

            nnet.save_params(epoch + 1)

        # terminating data fetching processes
        training_tasks.terminate()



class TrainUi(QMainWindow,Ui_TrainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.center()
        self.TrainBtn = self.toolButton_2
        self.ConfirmBtn = self.toolButton
        self.BackBtn = self.toolButton_3

        self.batchsize = self.lineEdit.text()
        self.batch_per_gpu = self.lineEdit_2.text()
        self.epochs = self.lineEdit_3.text()
        self.imgs_per_epoch = self.lineEdit_4.text()
        self.learning_rate = self.lineEdit_5.text()
        self.display = self.lineEdit_6.text()
        self.center_weight = self.lineEdit_7.text()
        self.scale_weight = self.lineEdit_8.text()
        self.offset_weight = self.lineEdit_9.text()
        self.start_epoch = self.lineEdit_10.text()
        self.seed = self.lineEdit_11.text()
        
        self.TrainBtn.clicked.connect(self.Train)
        self.ConfirmBtn.clicked.connect(self.Confirm)

        self.TrainThread = TrainThread()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2),
                  int((screen.height() - size.height()) / 2))

    def Confirm(self):
        self.batchsize = self.lineEdit.text()
        self.batch_per_gpu = self.lineEdit_2.text()
        self.epochs = self.lineEdit_3.text()
        self.imgs_per_epoch = self.lineEdit_4.text()
        self.learning_rate = self.lineEdit_5.text()
        self.display = self.lineEdit_6.text()
        self.center_weight = self.lineEdit_7.text()
        self.scale_weight = self.lineEdit_8.text()
        self.offset_weight = self.lineEdit_9.text()
        self.start_epoch = self.lineEdit_10.text()
        self.seed = self.lineEdit_11.text()
        messbox = QMessageBox()
        messbox.setWindowTitle(u'提示')
        messbox.setText(u'训练配置成功')
        messbox.exec_()

    def Train(self):
        self.TrainThread.batchsize = int(self.batchsize)
        self.TrainThread.batch_per_gpu = eval(self.batch_per_gpu)
        self.TrainThread.epochs = int(self.epochs)
        self.TrainThread.imgs_per_epoch = int(self.imgs_per_epoch)
        self.TrainThread.learning_rate = float(self.learning_rate)
        self.TrainThread.display = int(self.display)
        self.TrainThread.center_weight = float(self.center_weight)
        self.TrainThread.scale_weight = float(self.scale_weight)
        self.TrainThread.offset_weight = float(self.offset_weight)
        self.TrainThread.start_epoch = int(self.start_epoch)
        self.TrainThread.seed = int(self.seed)
        self.textBrowser.repaint()
        self.TrainThread.start()   
        # self.textBrowser.append("system config...")
        # self.textBrowser.append(f'Config:\n{cfg_file.pretty_text}')
        self.TrainThread.configsignal.connect(self.show_cfg)
        self.TrainThread.logsignal.connect(self.show_log)

    def show_log(self,log):
        self.textBrowser.append(log)
        self.textBrowser.repaint()

    def show_cfg(self,cfg):
        self.textBrowser.append(cfg)
        self.textBrowser.repaint()

        # if not os.path.exists(cfg_file.train_cfg.work_dir):
        #     os.makedirs(cfg_file.train_cfg.work_dir)

        # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # log_file = os.path.join(cfg_file.train_cfg.work_dir, f'{timestamp}.log')
        # logger = get_logger(name='CAP', log_file=log_file)
        # json_file = os.path.join(cfg_file.train_cfg.work_dir, f'{timestamp}.json')

        # cfg_file.dataset.batch_size = int(self.batchsize)
        # cfg_file.train_cfg.chunk_sizes = eval(self.batch_per_gpu)
        # cfg_file.train_cfg.num_epochs = int(self.epochs)
        # cfg_file.train_cfg.iter_per_epoch = int(self.imgs_per_epoch)
        # cfg_file.train_cfg.learning_rate = float(self.learning_rate)
        # cfg_file.train_cfg.display = int(self.display)
        # cfg_file.kp_head.csp_center_loss.loss_weight = float(self.center_weight)
        # cfg_file.kp_head.regr_h_loss.loss_weight = float(self.scale_weight)
        # cfg_file.kp_head.regr_offset_loss.loss_weight = float(self.offset_weight)
        # start_epoch = int(self.start_epoch)
        # seed = int(self.seed)

        # logger.info("system config...")
        # logger.info(f'Config:\n{cfg_file.pretty_text}')

        # self.textBrowser.append("system config...")
        # self.textBrowser.append(f'Config:\n{cfg_file.pretty_text}')        

        # set_seed(int(seed))

        # learning_rate    = cfg_file.train_cfg.learning_rate
        # pretrained_model = cfg_file.train_cfg.pretrain
        # display          = cfg_file.train_cfg.display
        # sample_module    = cfg_file.train_cfg.sample_module
        # iter_per_epoch   = cfg_file.train_cfg.iter_per_epoch
        # num_epochs       = cfg_file.train_cfg.num_epochs
        # batch_size       = cfg_file.dataset.batch_size

        # # queues storing data for training
        # training_queue   = Queue(cfg_file.train_cfg.prefetch_size)

        # # load data sampling function
        # data_file   = "sample.{}".format(sample_module)
        # sample_data = importlib.import_module(data_file).sample_data

        # if cfg_file.train_cfg.cache_ped:
        #     with open(cfg_file.train_cfg.cache_ped, 'rb') as fid:
        #         ped_data = pickle.load(fid)
        # if cfg_file.train_cfg.cache_emp:
        #     with open(cfg_file.train_cfg.cache_emp, 'rb') as fid:
        #         emp_data = pickle.load(fid)
        # length_dataset = len(ped_data)+len(emp_data)
        # logger.info('the length of dataset is: {}'.format(length_dataset))
        # self.textBrowser.append('the length of dataset is: {}'.format(length_dataset))

        # # allocating resources for parallel reading
        # if cfg_file.train_cfg.cache_emp:
        #     training_tasks   = init_parallel_jobs(cfg_file, training_queue, sample_data, ped_data, emp_data)
        # else:
        #     training_tasks = init_parallel_jobs(cfg_file, training_queue, sample_data, ped_data)
        # # prefetch_data(cfg, training_queue, sample_data, ped_data, emp_data)

        # logger.info("building model...")
        # self.textBrowser.append("building model...")

        # nnet = NetworkFactory(cfg_file)

        # if pretrained_model is not None:
        #     if not os.path.exists(pretrained_model):
        #         raise ValueError("pretrained model does not exist")
        #     logger.info("loading from pretrained model")
        #     nnet.load_pretrained_params(pretrained_model)

        # if start_epoch:
        #     nnet.load_params(start_epoch)
        #     nnet.set_lr(learning_rate)
        #     logger.info("training starts from iteration {} with learning_rate {}".format(start_epoch, learning_rate))
        #     self.textBrowser.append("training starts from iteration {} with learning_rate {}".format(start_epoch, learning_rate))
        # else:
        #     nnet.set_lr(learning_rate)

        # logger.info("training start...")
        # self.textBrowser.append("training start...")
        # self.textBrowser.repaint()
        # nnet.cuda()
        # nnet.train_mode()
        # epoch_length = int(iter_per_epoch / batch_size)
        # json_obj = open(json_file, 'w')
        # loss = []
        # for epoch in range(start_epoch, num_epochs):
        #     for iteration in range(1, epoch_length + 1):
        #         training = training_queue.get(block=True)
        #         training_loss = nnet.train(**training)

        #         loss.append(training_loss.item())

        #         if display and iteration % display == 0:
        #             loss = np.array(loss)
        #             logger.info("Epoch: {}/{}, loss_csp: {:.5f}".format(epoch+1, num_epochs, loss.sum() / display))
        #             self.textBrowser.append("Epoch: {}/{}, loss_csp: {:.5f}".format(epoch+1, num_epochs, loss.sum() / display))
        #             self.textBrowser.repaint()
        #             text = {"Epoch": epoch+1, "loss_csp": round(loss.sum() / display, 5)}
        #             text = json.dumps(text)
        #             json_obj.write(text)
        #             json_obj.write('\r\n')
        #             loss = []

        #         del training_loss

        #     nnet.save_params(epoch + 1)

        # # terminating data fetching processes
        # training_tasks.terminate()



if __name__ == '__main__':
    app = QApplication(sys.argv)

    Gui = TrainUi()
    Gui.show()

    app.exec()