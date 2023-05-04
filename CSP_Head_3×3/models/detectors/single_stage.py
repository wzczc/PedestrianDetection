import torch.nn as nn

from models.backbone import ResNet
from models.head import KpHead
from models.py_utils import parse_losses, parse_det_offset
import pdb


class SingleStageDetector(nn.Module):
    def __init__(self, cfg):
        super(SingleStageDetector, self).__init__()

        self.backbone = ResNet(**cfg.backbone)

        self.csp_head = KpHead(**cfg.kp_head)

        self.test = True if cfg.test_cfg.test else False
        self.cfg = cfg

    def forward(self, img, **kwargs):
        x = self.backbone(img)
        outs_csp = self.csp_head(x)

        if self.test:
            detections, head_detections = self.simple_test(outs_csp, **kwargs)
            return detections, head_detections
        else:
            return outs_csp

    def loss(self, preds, targets):
        preds_csp = preds
        seman_map, scale_map, offset_map, head_center_map,head_scale_map,head_offset_map = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5]
        losses_csp = self.csp_head.loss(preds_csp, seman_map=seman_map, scale_map=scale_map, offset_map=offset_map, head_center_map=head_center_map,head_scale_map=head_scale_map,head_offset_map=head_offset_map)
        loss_csp, loss_csp_var,loss_center,loss_scale,loss_offset,loss_head_center,loss_head_scale,loss_head_offset = parse_losses(losses_csp)
        return loss_csp,loss_center,loss_scale,loss_offset,loss_head_center,loss_head_scale,loss_head_offset

    def simple_test(self, outs, **kwargs):
        outs_csp = [outs[0].sigmoid(), outs[1], outs[2]]
        outs_head = [outs[3].sigmoid(), outs[4], outs[5]]
        nms_algorithm = {
            "nms": 0,
            "linear_soft_nms": 1,
            "exp_soft_nms": 2
        }[self.cfg.test_cfg.nms_algorithm]

        outs = []
        head_outs = []
        scores_csp = self.cfg.test_cfg.scores_csp
        scores_head = self.cfg.test_cfg.scores_head
        for out_csp in outs_csp:
            out_csp = out_csp.data.cpu().numpy()
            outs.append(out_csp)
        for out_head in outs_head:
            out_head = out_head.data.cpu().numpy()
            head_outs.append(out_head)
        dets, head_dets = parse_det_offset(outs, head_outs, self.cfg, nms_algorithm, score=scores_csp, score_head=scores_csp)

        return dets, head_dets
