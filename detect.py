# encoding=utf-8
import os

import cv2
from items import MessageItem
import time
import numpy as np

train_path = 'trainval/'
test_path = 'test_public/'
val_out = 'pre_val/'
test_out = 'pre_test/'

'''
监视者模块,负责入侵检测,目标跟踪
'''


class WatchDog(object):
    # 入侵检测者模块,用于入侵检测
    def __init__(self, frame=None):
        # 运动检测器构造函数
        self._background = None
        if frame is not None:
            self._background = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    def isWorking(self):
        # 运动检测器是否工作
        return self._background is not None

    def startWorking(self, frame):
        # 运动检测器开始工作
        if frame is not None:
            self._background = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

    def stopWorking(self):
        # 运动检测器结束工作
        self._background = None

    def analyze(self, frame):
        # 运动检测
        if frame is None or self._background is None:
            return
        sample_frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        diff = cv2.absdiff(self._background, sample_frame)
        diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, self.es, iterations=2)
        image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coordinate = []
        bigC = None
        bigMulti = 0
        for c in cnts:
            if cv2.contourArea(c) < 1500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            if w * h > bigMulti:
                bigMulti = w * h
                bigC = ((x, y), (x + w, y + h))
        if bigC:
            cv2.rectangle(frame, bigC[0], bigC[1], (255, 0, 0), 2, 1)
        coordinate.append(bigC)
        message = {"coord": coordinate, 'msg': None}
        return MessageItem(frame, message)


class Tracker(object):
    """
    追踪者模块,用于追踪指定目标
    """

    def __init__(self, tracker_type="BOOSTING", draw_coord=True):
        """
        初始化追踪器种类
        """
        # 获得opencv版本
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        # 构造追踪器
        if int(minor_ver) < 3:
            # self.tracker = cv2.Tracker_create(tracker_type)
            self.tracker = cv2.legacy.TrackerKCF_create()
            print(self.tracker)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.legacy.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()
            print(self.tracker)

    def initWorking(self, frame, box):
        """
        追踪器工作初始化
        frame:初始化追踪画面
        box:追踪的区域
        """
        if not self.tracker:
            raise Exception("追踪器未初始化")
        status = self.tracker.init(frame, box)
        if not status:
            raise Exception("追踪器工作初始化失败")
        self.coord = box
        self.isWorking = True

    def track(self, frame):
        """
        开启追踪
        """
        message = None
        lu = (0, 0)
        rd = (0, 0)
        if self.isWorking:
            status, self.coord = self.tracker.update(frame)
            if status:
                message = {"coord": [((int(self.coord[0]), int(self.coord[1])),
                                      (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3])))]}
                if self.draw_coord:
                    p1 = (int(self.coord[0]), int(self.coord[1]))
                    lu = (self.coord[0], self.coord[1])
                    p2 = (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3]))
                    rd = ((self.coord[0] + self.coord[2]), (self.coord[1] + self.coord[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    message['msg'] = "is tracking"
        return lu, rd, MessageItem(frame, message)


class ObjectTracker(object):
    def __init__(self, dataSet):
        self.cascade = cv2.CascadeClassifier(dataSet)

    def track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.03, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame


if __name__ == '__main__':
    a = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    # for dirs in os.listdir(test_path):
    for dirs in os.listdir(train_path):
        tracker = Tracker(tracker_type="KCF")

        # newdir = os.path.join(test_path, dirs)
        newdir = os.path.join(train_path, dirs)
        print('start processing ' + dirs)
        f = open(newdir + '/groundtruth.txt')
        line = f.readline()
        x1, y1, x2, y2, x3, y3, x4, y4 = [int(float(i)) for i in line.split(',')]
        img = cv2.imread(newdir + '/00000001.jpg')
        bbox = (x1, y1, x3, y3)
        # print(bbox)
        tracker.initWorking(img, bbox)
        p1 = (x1, y1)
        p2 = (x3, y3)
        with open(val_out + dirs + '.txt', 'w') as out:
        # with open(test_out + dirs + '.txt', 'w') as out:
            out.write(line)
            for names in os.listdir(newdir):
                if names == '00000001.jpg' or names == 'groundtruth.txt':
                    continue
                new_img = cv2.imread(newdir + '/' + names)
                left_up, right_down, item = tracker.track(new_img)
                if left_up != (0, 0) and right_down != (0, 0):
                    p1 = left_up
                    p2 = right_down
                # print(p1, p2)
                line = str(p1[0]) + ',' + str(p1[1]) + ',' + str(p2[0]) + ',' + str(p1[1]) + ',' + str(p2[0]) + ',' + \
                       str(p2[1]) + ',' + str(p1[0]) + ',' + str(p2[1]) + '\n'
                out.write(line)
                # cv2.imshow("track", item.getFrame())
                # cv2.waitKey(0)
                # break
        out.close()
        f.close()
        print('end processing ' + dirs)
        # break

    # video = cv2.VideoCapture(0)
    # ok, frame = video.read()
    # bbox = cv2.selectROI(frame, False)
    # print(bbox)
    # tracker.initWorking(frame, bbox)
    # while True:
    #     _, frame = video.read()
    #     if _:
    #         item = tracker.track(frame)
    #         cv2.imshow("track", item.getFrame())
    #         k = cv2.waitKey(1) & 0xff
    #         if k == 27:
    #             break
