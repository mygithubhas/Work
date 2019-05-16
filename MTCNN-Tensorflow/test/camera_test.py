#coding:utf-8
import sys
sys.path.append('H:/homework/MTCNN-Tensorflow')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np

test_mode = "onet"#测试模型（模式）
thresh = [0.9, 0.6, 0.7]#翻滚
min_face_size = 24#最小脸部形状
stride = 2#步长
slide_window = False#滑动窗口
shuffle = False#拖拽
#vis = True
detectors = [None, None, None]#初始化探测器
#模型路径
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]#文件后缀名
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]#zip将列表中每个队列位置相同的元素打包成元组
PNet = FcnDetector(P_Net, model_path[0])#定义Pnet网络运算
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])#定义RNet网络运算
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])#定义ONet网络运算
detectors[2] = ONet
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)#定义网络运算

video_capture = cv2.VideoCapture(1)
#video_capture.set(3, 1920)
#video_capture.set(4, 1080)
corpbbox = None
while True:
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()#记录时间有t1
    ret, frame = video_capture.read()#读取帧数图片返回ture和图片
    if ret:
        image = np.array(frame)#转化为numpy
        boxes_c,landmarks = mtcnn_detector.detect(image)#检测图片返回切片后的图，

        print(landmarks.shape)#显示图片形状
        t2 = cv2.getTickCount()#记录时间t2
        t = (t2 - t1) / cv2.getTickFrequency()
        fps = 1.0 / t#计算帧数
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # if score > thresh:
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)#画矩形
            cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)#写字
        cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)#写字
        for i in range(landmarks.shape[0]):#画圈
            for j in range(len(landmarks[i])//2):
                cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))            
        # time end
        #frame=cv2.resize(frame,(1920,1080))#改变图片大小
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('device not find')
        break
video_capture.release()#关闭图片
cv2.destroyAllWindows()#关闭窗口
