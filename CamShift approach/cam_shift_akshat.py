#This code uses the CamShift algorithm to track any object selected by the user during run time.
#Usage: Run the code using python cam_shift_akshat.py.
#The camera feed will open up. Select the object you want to0 track using the mouse (click, drag, release using left click).
#You may press 'p' to pause the frame if you cannot select the object with live feed.
#You may press 'd' to finalise the selection
#You may press 'r' to reset and select again

import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator

#Create a VideoCapture object to get live feed from webcam
cam = cv2.VideoCapture(0)

#Initialize Variables
refPt = []
track_window = ()
Mode = 0
ranges = [[0, 180], [0, 255], [0, 255]]
thresh = [0, 0, 0]
kernel = np.ones((3,3),np.uint8)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

#Function gets RoI when user clicks and drags on object to be tracked using the mouse
def get_roi(event, x, y, flags, param):
    global refPt, track_window

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))

        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow('Camshift', img)

cv2.namedWindow('Camshift')
cv2.setMouseCallback('Camshift', get_roi)

while(1):
    #Read the incoming camera frame
    ret, img = cam.read()

    #Add median blur
    img = cv2.medianBlur(img,5)

    #Duplicate for backup
    duplicate = img.copy()

    #Convert to HSV
    hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    #Mode = 0 until the user doesn't select an object
    if Mode == 0:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'): #pause to select object
            key = cv2.waitKey(0) & 0xFF
        if key == ord('r'): #Reset
            img = duplicate.copy()
            refPt = []
        elif key == ord('d'): #Done selecting
            Mode = 1

            #Define Region of Interest
            roi = duplicate[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            cv2.imshow('ROI', roi)

            #Initialize a track_window value
            track_window = (refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1])

            hsv = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2HSV)

            #Get the Histogram, normalize and get the dynamic threshold value
            for i in range(3):
                histr = cv2.calcHist([hsv],[i],None,[ranges[i][1]],ranges[i])
                if i==0:
                    histrr = histr
                cv2.normalize(histr, histr, ranges[i][0], ranges[i][1], cv2.NORM_MINMAX);

                thresh[i], max_value = max(enumerate(histr), key=operator.itemgetter(1))

    if Mode == 1:
        #Creating arrays to store the min and max HSV ranges
        lower = np.array([thresh[0]-30, thresh[1]-30, thresh[2]-40])
        upper = np.array([thresh[0]+30, thresh[1]+30, thresh[2]+40])

        #Create binary mask of in range object
        mask = cv2.inRange(hsv_img, lower, upper)

        #Morphology
        erosion = cv2.erode(mask,kernel,iterations = 2)
        dilation = cv2.dilate(erosion,kernel,iterations = 3)
        opening = cv2.dilate(dilation,kernel,iterations = 3)

        #Calculate Back Propogation
        dst = cv2.calcBackProject([hsv_img],[0],histrr,[0,180],1)

        #CamShift
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        duplicate[opening == 0] = 0

        if ret[1][0] >= ret[1][1]:
            ret = (ret[0], (abs(ret[1][0]*1.5), abs(ret[1][1]*1.3)), ret[2])
        else:
            ret = (ret[0], (abs(ret[1][0]*1.3), abs(ret[1][1]*1.5)), ret[2])

        #Draw an ellipse
        cv2.ellipse(img, ret, (0, 0, 255), 2)

        cv2.imshow('Object', opening)
        k = cv2.waitKey(1) & 0xFF
        if(k==27):
            break
    cv2.imshow('Camshift', img)
