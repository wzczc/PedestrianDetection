import numpy as np


def calc_gt_center(size_train, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
        s = 2*(sigma**2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx,(-1,1))
    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    scale_map = np.zeros((2, int(size_train[0]/down), int(size_train[1]/down)))
    if scale=='hw':
        scale_map = np.zeros((3, int(size_train[0] / down), int(size_train[1] / down)))
    if offset:
        offset_map = np.zeros((3, int(size_train[0] / down), int(size_train[1] / down)))
    seman_map = np.zeros((3, int(size_train[0]/down), int(size_train[1]/down)))
    seman_map[1,:,:] = 1
    head_center_map = np.zeros((3, int(size_train[0]/down), int(size_train[1]/down)))
    head_center_map[1,:,:] = 1
    head_scale_map = np.zeros((2, int(size_train[0]/down), int(size_train[1]/down)))
    head_offset_map = np.zeros((3, int(size_train[0] / down), int(size_train[1] / down)))
    if len(igs) > 0:
        igs = igs/down
        for ind in range(len(igs)):
            x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
            seman_map[1, y1:y2, x1:x2] = 0

            x3,y3,x4,y4 = int(x1+0.33*(x2-x1)), y1, int(x2-0.33*(x2-x1)), int(y1+0.15*(y2-y1))
            head_center_map[1, y3:y4, x3:x4] = 0
    if len(gts)>0:
        gts = gts/down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2-x1)
            dy = gaussian(y2-y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_map[0, y1:y2, x1:x2] = np.maximum(seman_map[0, y1:y2, x1:x2], gau_map)
            seman_map[1, y1:y2, x1:x2] = 1
            seman_map[2, c_y, c_x] = 1

            x3, y3, x4, y4 = int(x1+0.33*(x2-x1)), y1, int(x2-0.33*(x2-x1)), int(y1+0.15*(y2-y1))
            c_x2, c_y2 = int((gts[ind, 0] + gts[ind, 2]) / 2), int((2*gts[ind, 1]+0.15*(gts[ind, 3] - gts[ind, 1])) / 2)
            dx2 = gaussian(x4-x3)
            dy2 = gaussian(y4-y3)
            gau_map2 = np.multiply(dy2, np.transpose(dx2))
            head_center_map[0, y3:y4, x3:x4] = np.maximum(head_center_map[0, y3:y4, x3:x4], gau_map2)
            head_center_map[1, y3:y4, x3:x4] = 1
            head_center_map[2, c_y2, c_x2] = 1

            if scale == 'h':
                scale_map[0, c_y-r:c_y+r+1, c_x-r:c_x+r+1] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[1, c_y-r:c_y+r+1, c_x-r:c_x+r+1] = 1
            elif scale=='w':
                scale_map[0, c_y-r:c_y+r+1, c_x-r:c_x+r+1] = np.log(gts[ind, 2] - gts[ind, 0])
                scale_map[1, c_y-r:c_y+r+1, c_x-r:c_x+r+1] = 1
            elif scale=='hw':
                scale_map[0, c_y-r:c_y+r+1, c_x-r:c_x+r+1] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[1, c_y-r:c_y+r+1, c_x-r:c_x+r+1] = np.log(gts[ind, 2] - gts[ind, 0])
                scale_map[2, c_y-r:c_y+r+1, c_x-r:c_x+r+1] = 1
            if offset:
                offset_map[0, c_y, c_x] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[1, c_y, c_x] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[2, c_y, c_x] = 1

            head_scale_map[0, c_y2-r:c_y2+r+1, c_x2-r:c_x2+r+1] = np.log(0.15*(gts[ind, 3] - gts[ind, 1]))
            head_scale_map[1, c_y2-r:c_y2+r+1, c_x2-r:c_x2+r+1] = 1
            head_offset_map[0, c_y2, c_x2] = (2*gts[ind, 1]+0.15*(gts[ind, 3] - gts[ind, 1])) / 2 - c_y2 - 0.5
            head_offset_map[1, c_y2, c_x2] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x2 - 0.5
            head_offset_map[2, c_y2, c_x2] = 1

    if offset:
        return np.array(seman_map, dtype=np.float32),np.array(scale_map, dtype=np.float32),np.array(offset_map, dtype=np.float32),\
               np.array(head_center_map, dtype=np.float32),np.array(head_scale_map, dtype=np.float32),np.array(head_offset_map, dtype=np.float32)
    else:
        return seman_map, scale_map
