from Detect_objects import DetectObjectYOLOV3


detect = DetectObjectYOLOV3('rtsp://admin:AdminNLT!1@192.168.254.18:554?tcp', 0.4, 0.5)
detect.connect()
detect.detect()
