from tensorflow.keras.models import load_model
import cv2
# import pandas as pd
import numpy as np

def findcolor(x,fls,sr):
    for cnt in x:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 1000:
            a, b, c, d = cv2.boundingRect(cnt)
            cv2.rectangle(fls, (a, b), (a + c, b + d), (0, 0, 255), 2)
            cv2.putText(fls, sr, (a, b-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


mdl = load_model('Model.h5')
mdl.summary()

# Below function will read video imgs
cap = cv2.VideoCapture(0)

while 1:
    _, fls = cap.read()
    fls = cv2.resize(fls, (640, 480))
    # Make a copy to draw contour outline
    hsv = cv2.cvtColor(fls, cv2.COLOR_BGR2HSV)

    # find contours in the red mask
    contours_red, _ = cv2.findContours(
        cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the green mask
    contours_green, _ = cv2.findContours(
        cv2.inRange(hsv, np.array([40, 20, 50]), np.array([90, 255, 255])), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the blue mask
    contours_blue, _ = cv2.findContours(
        cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255])), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the yellow mask
    contours_yellow, _ = cv2.findContours(
        cv2.inRange(hsv, np.array([25, 70, 120]), np.array([30, 255, 255])), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # loop through the red contours and draw a rectangle around them

    findcolor(contours_red,fls,'red')
    findcolor(contours_green,fls,'green')
    findcolor(contours_blue,fls,'blue')
    findcolor(contours_yellow,fls,'yellow')

    cv2.imshow('Color Recognition', fls)

    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
