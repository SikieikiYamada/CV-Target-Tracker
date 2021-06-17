# encoding=utf-8
import time
import numpy
import base64
import os
import logging
import sys
import struct
# from setting import *
from scrapy import settings
from PIL import Image
from io import BytesIO


# 工具类
class IOUtil(object):
    # 流操作工具类
    @staticmethod
    def array_to_bytes(pic, formatter="jpeg", quality=70):
        """
        静态方法,将numpy数组转化二进制流
        :param pic: numpy数组
        :param format: 图片格式
        :param quality:压缩比,压缩比越高,产生的二进制数据越短
        :return:
        """
        stream = BytesIO()
        picture = Image.fromarray(pic)
        picture.save(stream, format=formatter, quality=quality)
        jepg = stream.getvalue()
        stream.close()
        return jepg

    @staticmethod
    def bytes_to_base64(byte):
        """
        静态方法,bytes转base64编码
        :param byte:
        :return:
        """
        return base64.b64encode(byte)

    @staticmethod
    def transport_rgb(frame):
        """
        将bgr图像转化为rgb图像,或者将rgb图像转化为bgr图像
        """
        return frame[..., ::-1]

    @staticmethod
    def byte_to_package(bytes, cmd, var=1):
        """
        将每一帧的图片流的二进制数据进行分包
        :param bytes: 二进制文件
        :param cmd:命令
        :return:
        """
        head = [var, len(bytes), cmd]
        headPack = struct.pack("!3I", *head)
        senddata = headPack + bytes
        return senddata

    @staticmethod
    def mkdir(filePath):
        """
        创建文件夹
        """
        if not os.path.exists(filePath):
            os.mkdir(filePath)

    @staticmethod
    def countCenter(box):
        """
        计算一个矩形的中心
        """
        return int(abs(box[0][0] - box[1][0]) * 0.5) + box[0][0], int(abs(box[0][1] - box[1][1]) * 0.5) + box[0][1]

    @staticmethod
    def countBox(center):
        """
        根据两个点计算出,x,y,c,r
        """
        return center[0][0], center[0][1], center[1][0] - center[0][0], center[1][1] - center[0][1]

    @staticmethod
    def getImageFileName():
        return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.png'


# # 构造日志
# LOG_NAME = 'logs'
# LOG_FORMATTER = "%(asctime)s %(levelname)d %(message)s"
# DATEFMT = "[%Y_%m_%d_%H_%M_%S]"
# LOG_DIR = 'log/'
#
# logger = logging.getLogger(__name__)
# formatter = logging.Formatter(LOG_FORMATTER, DATEFMT)
# IOUtil.mkdir(LOG_DIR)
# file_handler = logging.FileHandler(LOG_DIR + LOG_FILE, encoding='utf-8')
# file_handler.setFormatter(formatter)
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.addHandler(console_handler)
# logger.setLevel(logging.INFO)
