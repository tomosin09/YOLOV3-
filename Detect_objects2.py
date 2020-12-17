from Adressing import Address
from Stream import VideoStream
import sys
import time
import cv2 as cv
import numpy as np

addresses = Address('rtsp://admin:AdminNLT!1@192.168.254.18:554?tcp', 'rtsp://admin:AdminNLT!1@192.168.254.18:554?tcp', 'rtsp://admin:AdminNLT!1@192.168.254.18:554?tcp', 'rtsp://admin:AdminNLT!1@192.168.254.18:554?tcp')
# # addresses = Address('1', '2')
# addresses = Address('rtsp://admin:AdminNLT!1@192.168.254.18:554?tcp')
cameras = addresses.getAddress()


class DetectObjects:
    def __init__(self, cameras):
        self.cameras = cameras
        self.streams = []
        self.frame = None

    # function to initialize some stream
    def connect(self):
        for i in self.cameras:
            print(i)
            stream = VideoStream(i).start()
            if stream.grabbed == 0:
                print('No frame')
            self.streams.append(stream)
            print(self.streams)
        cv.namedWindow("Stream", cv.WINDOW_NORMAL)
        time.sleep(1)

    def stop(self):
        for i in self.streams:
            i.stop()

    def getSize(self, image):
        h, w = image.shape[:2]
        return h, w

    # function to start VideoStream
    def video(self):
        while 1:
            arrayFrames = []
            for stream in self.streams:
                frame = stream.read()
                arrayFrames.append(frame)
            if cv.waitKey(10) != cv.waitKey(27):
                break
            if len(arrayFrames) == 1:
                self.frame = arrayFrames[0]
            if len(arrayFrames) == 2:
                self.frame = np.concatenate((arrayFrames[0], arrayFrames[1]), 0)
            if len(arrayFrames) == 3:
                h, w = self.getSize(arrayFrames[0])
                blackImage = np.zeros((h,w,3),dtype=np.uint8)
                frame1 = np.concatenate((arrayFrames[0], arrayFrames[1]), 0)
                frame2 = np.concatenate((arrayFrames[2], blackImage), 0)
                self.frame = np.concatenate((frame1,frame2),1)
            if len(arrayFrames) == 4:
                frame1 = np.concatenate((arrayFrames[0], arrayFrames[1]), 0)
                frame2 = np.concatenate((arrayFrames[2], arrayFrames[3]), 0)
                self.frame = np.concatenate((frame1, frame2), 1)
            cv.imshow('Stream', self.frame)
        cv.destroyAllWindows()
        self.stop()


Detection = DetectObjects(cameras)
Detection.connect()
Detection.video()
