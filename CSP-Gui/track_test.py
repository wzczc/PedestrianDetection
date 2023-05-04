import cv2
import sys

# 创建一个跟踪器，algorithm: KCF、CSRT、DaSiamRPN、GOTURM、MIL
trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

# # 创建一个跟踪器，algorithm: KCF、CSRT、DaSiamRPN、GOTURM、MIL
# tracker_types = ['MIL', 'KCF', 'CSRT', 'DaSiamRPN', 'GOTURM']
# def createTypeTracker(type):
#     if type == tracker_types[0]:
#         tracker = cv2.TrackerMIL_create()
#     elif type == tracker_types[1]:
#         tracker = cv2.TrackerKCF_create()
#     elif type == tracker_types[2]:
#         tracker = cv2.TrackerCSRT_create()
#     elif type == tracker_types[3]:
#         tracker = cv2.TrackerDaSiamRPN_create()
#     elif type == tracker_types[4]:
#         tracker = cv2.TrackerGOTURN_create()
#     else:
#         tracker = None

#     return tracker

# videoPth = './VCG42503745058.mp4'
# if __name__ == '__main__':
#     tracker_type = 'MIL'
#     tracker = createTypeTracker(tracker_type)
#     # 读取视频
#     cap = cv2.VideoCapture(videoPth)
#     # 第一帧
#     ret, firstFrame = cap.read()
#     # 在第一帧中选取跟踪区域
#     box = cv2.selectROI('select ROI @1st Frame', firstFrame, fromCenter=True)
#     print(box)

#     # 初始化跟踪器
#     ok = tracker.init(firstFrame, box)
#     # 按帧读取视频
#     while cap:
#         ret, frame = cap.read()
#         if not ret:
#             print('read video error!')
#             break
#         # 计时器
#         timer = cv2.getTickCount()
#         ok, box = tracker.update(frame)
#         # print(box)
#         # box=(x,y,h,w) 为一个四元素元组，前两个为矩形的左上角顶点坐标，后两个为矩形的尺寸
#         # 计算帧率
#         fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)     
#         if ok:
#             # 画出矩形目标区域
#             pt1 = (int(box[0]), int(box[1]))
#             pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
#             cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2, 1)
#             cv2.imshow('track',frame)
#             cv2.waitKey(1)
#         else:
#             # 显示跟踪失败
#             cv2.putText(frame, 'track failed!', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0))



def createTypeTracker(trackerType):
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]: # 暂时存在问题
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None

    return tracker


pth = './VCG42503745058.mp4'
cap = cv2.VideoCapture(pth)
success, first_frame = cap.read()
if not success:
    print('opening video failed!')
    sys.exit(1)
# h, w = frame.shape[:2]
# # print(frame.shape[:2])
# frame_resize = cv2.resize(frame, (int(w / 4), int(h / 4)))
# 在C + + 版本中，selectROI允许您获取多个边界框，但在Python版本中，它只返回一个边界框。
boxs = []
for i in range(2):
    boxs.append(cv2.selectROI('select ROI',first_frame))
    # cv2.rectangle(frame,p1,p2,(0,0,255),2)
print(boxs)
tracker = cv2.MultiTracker_create()
for box in boxs:
    if box[0] or box[1] or box[2] or box[3]:
        tracker.add(createTypeTracker('BOOSTING'), first_frame, box)  
while success:
    ret, frame = cap.read()
    # 原图像为4K分辨率，这里降采样适应电脑屏幕，视情况而定
    # dsize_frame = cv2.resize(frame, (int(w / 4), int(h / 4)))
    ok, boxs = tracker.update(frame)
    if len(boxs) > 0:
        for box in boxs:
            # 画出矩形区域
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2, 1)
            cv2.imshow('img',frame)
            if cv2.waitKey(1) == 'q':
                break
    else:
        # 显示跟踪失败
        cv2.putText(frame, 'track failed!', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0))



