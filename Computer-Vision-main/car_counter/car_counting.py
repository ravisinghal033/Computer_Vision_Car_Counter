from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy

# cap = cv2.VideoCapture(0) # for webcam
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("../videos/cars.mp4")


model = YOLO("yolov5n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (1280, 720))

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCounts = []


while True:
    succss, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)

            w, h = x2-x1, y2-y1
            # x1, y1, w, h = box.xywh[0]
            # cvzone.cornerRect(img, (x1, y1, w, h), l=8)

            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            #class names
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus' or currentClass == 'motorbike' and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(20, y1)), scale=0.6, offset=5,
                #                    thickness=1)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections =  np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)

    for results in resultsTracker:
        x1,y1,x2,y2,id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(results)
        w,h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(20, y1)), scale=2, offset=10,
                           thickness=3)

        cx, cy = x1+w//2, y1+h//2
        # cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if limits[0]<cx< limits[2] and limits[1]-15<cy<limits[1]+15:
            if totalCounts.count(id) == 0:
                totalCounts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f' Count: {len(totalCounts)}', (50, 50))

    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)