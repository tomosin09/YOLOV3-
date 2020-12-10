import cv2 as cv
import numpy as np
import time
from Stream import VideoStream


# 'rtsp://admin:AdminNLT!1@192.168.254.18:554?tcp'

class DetectObjectYOLOV3:

    def __init__(self, address, confid, nms):
        self.address = address
        # values for filtering out bbox
        self.confid = confid
        self.nms = nms
        # Path to coco.names
        self.classesFile = "coco.names"
        # Value for array of class names
        self.classes = None
        # Paths to .cfg and .weights
        self.Config = 'data/yolov3.cfg'
        self.Weights = 'data/yolov3.weights'
        # Reading a network model stored in Darknet model files.
        self.net = cv.dnn.readNetFromDarknet(self.Config, self.Weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        # Selecting a target device for computing
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        self.stream = None
        self.frame = None
        self.outs = None
        # Getting values name classes
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def connect(self):
        # Video stream initialisation
        self.stream = VideoStream(self.address).start()
        time.sleep(1)
        cv.namedWindow("Stream", cv.WINDOW_NORMAL)

    def getOutputsNames(self, net):
        # Get the name of all layers of the network
        layersNames = net.getLayerNames()
        # Get the index of the output layers
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Painting bboxes
    def drawPred(self, classId, conf, left, top, right, bottom):
        cv.rectangle(self.frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        # Get a label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)
        # Displaying the label at the top of the bbox
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(self.frame, (left, top - round(1.5 * labelSize[1])),
                     (left + round(1.5 * labelSize[0]), top + baseLine),
                     (255, 255, 255), cv.FILLED)
        cv.putText(self.frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Perform sorting bboxes
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confid:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        # Performs non maximum suppression given boxes and corresponding scores.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confid, self.nms)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    def detect(self):
        while 1:
            self.frame = self.stream.read()
            if cv.waitKey(10) != cv.waitKey(27):
                break
            blob = cv.dnn.blobFromImage(self.frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.getOutputsNames(self.net))
            self.postprocess(self.frame, outs)
            t, _ = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(self.frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv.imshow('Stream', self.frame)

        cv.destroyAllWindows()
        self.stream.stop()
