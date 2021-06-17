# encoding=utf-8
import json
from utils import IOUtil

'''
信息封装类
'''


class MessageItem(object):
    # 用于封装信息的类,包含图片和其他信息
    def __init__(self, frame, message):
        self._frame = frame
        self._message = message

    def getFrame(self):
        # 图片信息
        return self._frame

    def getMessage(self):
        # 文字信息,json格式
        return self._message

    def getBase64Frame(self):
        # 返回base64格式的图片,将BGR图像转化为RGB图像
        jepg = IOUtil.array_to_bytes(self._frame[..., ::-1])
        return IOUtil.bytes_to_base64(jepg)

    def getBase64FrameByte(self):
        # 返回base64格式图片的bytes
        return bytes(self.getBase64Frame())

    def getJson(self):
        # 获得json数据格式
        dicdata = {"frame": self.getBase64Frame().decode(), "message": self.getMessage()}
        return json.dumps(dicdata)

    def getBinaryFrame(self):
        return IOUtil.array_to_bytes(self._frame[..., ::-1])
