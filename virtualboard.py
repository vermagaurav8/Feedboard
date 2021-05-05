import cv2
import numpy as np
import time
import os
import handTrackingModule as htm

#########################

brushThickness = 20
eraserThickness = 70

#########################

folderPath = "Header"
my_list = os.listdir(folderPath)

overlay_list = []
for image_path in my_list:
    image = cv2.imread(f'{folderPath}/{image_path}')
    overlay_list.append(image)

header = overlay_list[0]
drawColor = (173, 74, 0)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # set 1280*720 px
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0

imageCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    landmark_list = detector.findPosition(img, draw=False)

    if len(landmark_list) != 0:

        # tip of index & middle finger
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        # 3. Checking which fingers are up
        fingures = detector.fingersUp()

        # 4. Two fingers mode : Selection Mode
        if fingures[1] and fingures[2]:

            xp, yp = 0, 0

            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlay_list[0]
                    drawColor = (173, 74, 0)
                elif 550 < x1 < 750:
                    header = overlay_list[1]
                    drawColor = (77, 145, 255)
                elif 800 < x1 < 950:
                    header = overlay_list[2]
                    drawColor = (22, 22, 255)
                elif 1050 < x1 < 1200:
                    header = overlay_list[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor,
                          cv2.FILLED)

        # 5. Drawing Mode
        if fingures[1] and fingures[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor,
                         eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor,
                         brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imageCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imageCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imageCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
