import cv2 as cv
from threading import Thread


# Class was taken from imutils by Adrian Rosebrock
# https://github.com/jrosebr1/imutils/

class VideoStream:
    def __init__(self, src=0, name='VideoStream'):
        # Video stream initialization
        self.stream = cv.VideoCapture(src)
        # The first variable returns True if frame is not None
        (self.grabbed, self.frame) = self.stream.read()

        # Thread name initialization
        self.name = name

        # Initializing a variable to stop the thread
        self.stopped = False

    # Function to start thread
    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while 1:
            if self.stopped:
                return
            # read next frames if stop is False
            (self.grabbed, self.frame) = self.stream.read()

    # Function to get frame from VideoStream
    def read(self):
        return self.frame

    # Function to stop thread
    def stop(self):
        self.stopped = True
