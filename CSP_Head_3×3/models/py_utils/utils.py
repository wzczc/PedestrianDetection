import torch
import numpy as np
from collections import OrderedDict
from external import NMS
import pdb



def parse_losses(losses):
    log_vars = OrderedDict()
    # pdb.set_trace()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    loss_cls = sum(_value for _key, _value in log_vars.items() if 'loss_cls' in _key)
    loss_regr = sum(_value for _key, _value in log_vars.items() if 'loss_regr' in _key)
    loss_offset = sum(_value for _key, _value in log_vars.items() if 'loss_offset' in _key)
    loss_cls2 = sum(_value for _key, _value in log_vars.items() if 'loss_head_center' in _key)
    loss_regr2 = sum(_value for _key, _value in log_vars.items() if 'loss_head_regr' in _key)
    loss_offset2 = sum(_value for _key, _value in log_vars.items() if 'loss_head_offset' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars,loss_cls,loss_regr,loss_offset,loss_cls2,loss_regr2,loss_offset2


def parse_det_offset(Y, H, cfg, nms_algorithm, score=0.1, score_head=0.1, down=4):
    seman = Y[0][0, 0, :, :]
    height = Y[1][0, 0, :, :]
    offset_y = Y[2][0, 0, :, :]
    offset_x = Y[2][0, 1, :, :]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, cfg.dataset.size_test[1]), min(y1 + h, cfg.dataset.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = NMS(boxs, cfg.test_cfg.nms_threshold, nms_algorithm)
        # keep = py_cpu_nms(boxs, cfg.test_cfg.nms_threshold)
        boxs = boxs[keep, :]

    head_boxs = []
    # head = H[0][0, 0, :, :]
    # head_h = H[1][0, 0, :, :]
    # head_oy = H[2][0, 0, :, :]
    # head_ox = H[2][0, 1, :, :]
    # head_yc, head_xc = np.where(head > score_head)
    # head_boxs = []
    # if len(head_yc) > 0:
    #     for i in range(len(head_yc)):
    #         h = np.exp(head_h[head_yc[i], head_xc[i]]) * down
    #         w = 0.92*h
    #         o_y = head_oy[head_yc[i], head_xc[i]]
    #         o_x = head_ox[head_yc[i], head_xc[i]]
    #         s = head[head_yc[i], head_xc[i]]
    #         x1, y1 = max(0, (head_xc[i] + o_x + 0.5) * down - w / 2), max(0, (head_yc[i] + o_y + 0.5) * down - h / 2)
    #         head_boxs.append([x1, y1, min(x1 + w, cfg.dataset.size_test[1]), min(y1 + h, cfg.dataset.size_test[0]), s])
    #     head_boxs = np.asarray(head_boxs, dtype=np.float32)
    #     head_keep = NMS(head_boxs, cfg.test_cfg.nms_threshold, nms_algorithm)
    #     # keep = py_cpu_nms(boxs, cfg.test_cfg.nms_threshold)
    #     head_boxs = head_boxs[head_keep, :]

    # bboxs = []
    # for box in boxs:
    #     maxiou = 0
    #     for head_box in head_boxs:
    #         hiou = HIoU(head_box,box)
    #         if hiou > maxiou:
    #             maxiou = hiou  
    #             head_bbox = head_box   
    #     if maxiou <= 0.5:
    #         if maxiou < 0.35:
    #             box[4] = max(0,box[4]-(0.5-maxiou))
    #         else:
    #             box[4] = max(0,box[4]-(0.5-maxiou)*(1-head_bbox[4]*1.6))
    #     else:
    #         box[4] = min(1,box[4]+(maxiou-0.5)*head_bbox[4]*0.9)
    #     if box[4] > score:
    #         bboxs.append(box)    

    # bboxs = np.asarray(bboxs, dtype=np.float32)
    # for head_box in head_boxs:
    #     miniou = 1
    #     for box in boxs:
    #         hiou = HIoU(head_box,box)
    #         if hiou < miniou:
    #             miniou = hiou
    #     if miniou < 0.1 and head_box[4] > 0.5:
    #         w = (head_box[2] - head_box[0])/0.34
    #         h = (head_box[3] - head_box[1])/0.15
    #         x1 = head_box[0] - 0.33*w
    #         y1 = head_box[1]
    #         x2 = head_box[2] + 0.33*w
    #         y2 = head_box[1] + h
    #         s = head_box[4]
    #         boxs = np.append(boxs,[[max(x1,0), y1, min(x2, cfg.dataset.size_test[1]), min(y2, cfg.dataset.size_test[0]), s]],axis=0)
        

    # i = 0
    # while i < len(boxs):
    #     max_iou = 0
    #     for head_box in head_boxs:
    #         hiou = HIoU(head_box,boxs[i])
    #         if hiou>max_iou:
    #             max_iou=hiou
    #     if max_iou < 0.75:
    #         boxs = np.delete(boxs,i,axis=0)
    #     else:
    #         i = i + 1

    return boxs , head_boxs


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def HIoU(head_box,box):  
    """
    计算头部框与全身框上部分的交集/头部框。
    :param rec1: (x0,y0,x1,y1)  头部框
    :param rec2: (x0,y0,x1,y1)  全身框
    """
    rec1 = head_box
    rec2 = box
    w = rec2[2] - rec2[0]
    h = rec2[3] - rec2[1]
    x1 = rec2[0] + 0.3*w 
    x2 = rec2[2] - 0.3*w 
    y1 = rec2[1] 
    y2 = rec2[1] + 0.18*h 

    left_column_max  = max(rec1[0],x1)
    right_column_min = min(rec1[2],x2)
    up_row_max       = max(rec1[1],y1)
    down_row_min     = min(rec1[3],y2)
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/S1