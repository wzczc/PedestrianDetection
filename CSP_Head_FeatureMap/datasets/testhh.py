import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

root_dir = '/home/lyw/wzc/CSP/CSP-pedestrian-pytorch-master/Caltech/train'
# root_dir = 'data/caltech/test'
all_img_path = os.path.join(root_dir, 'IMG')
all_anno_path = os.path.join(root_dir, 'anno_train10x_alignedby_RotatedFilters/')
res_path_gt = '/home/lyw/wzc/CSP/CSP-pedestrian-pytorch-master/data/cache/caltech/train_gt'
res_path_nogt = '/home/lyw/wzc/CSP/CSP-pedestrian-pytorch-master/data/cache/caltech/train_nogt'

rows, cols = 480, 640
image_data_gt, image_data_nogt = [], []

valid_count = 0
iggt_count = 0
box_count = 0
files = all_anno_path + 'set04_V002_I01352.txt'

boxes = []
ig_boxes = []
with open(files, 'rb') as fid:
	lines = fid.readlines()
print(len(lines))
if len(lines)>1:
	for i in range(1, len(lines)):
		info = lines[i].strip().split(' '.encode())
		label = info[0]
		occ, ignore = info[5], info[10]
		x1, y1 = max(int(float(info[1])), 0), max(int(float(info[2])), 0)
		w, h = min(int(float(info[3])), cols - x1 - 1), min(int(float(info[4])), rows - y1 - 1)
		box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
		if int(ignore) == 0:
			boxes.append(box)
		else:
			ig_boxes.append(box)
else:
	boxes = []
	ig_boxes = []

boxes = np.array(boxes)
ig_boxes = np.array(ig_boxes)

annotation = {}
box_count += len(boxes)
iggt_count += len(ig_boxes)
annotation['bboxes'] = boxes
annotation['ignoreareas'] = ig_boxes
if len(boxes) == 0:
	image_data_nogt.append(annotation)
else:
	image_data_gt.append(annotation)
	valid_count += 1
print('{} images and {} valid images, {} valid gt and {} ignored gt'.format(len(files), valid_count, box_count, iggt_count))
