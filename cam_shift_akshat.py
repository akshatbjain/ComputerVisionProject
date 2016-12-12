import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator

cam = cv2.VideoCapture(0)
refPt = []
track_window = ()
Mode = 0
ranges = [[0, 180], [0, 255], [0, 255]]
thresh = [0, 0, 0]
kernel = np.ones((3,3),np.uint8)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

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
    ret, img = cam.read()
    cv2.imshow('No filter', img)
    #img = cv2.bilateralFilter(img,9,75,75)
    img = cv2.medianBlur(img,5)
    duplicate = img.copy()
    hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    
    if Mode == 0:

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            cv2.waitKey(0)

        elif key == ord('d'):
            break

        elif key == ord('r'):
            img = duplicate.copy()


    if ((len(refPt) == 2) and (Mode == 0)):
        Mode = 1
        roi = duplicate[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow('ROI', roi)

        hsv = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2HSV)
        
        color = ['b', 'g', 'r']

        for i in range(3):
            histr = cv2.calcHist([hsv],[i],None,[ranges[i][1]],ranges[i])
            print histr.shape
            if i==0:
                histrr = histr
            cv2.normalize(histr, histr, ranges[i][0], ranges[i][1], cv2.NORM_MINMAX);

            #min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
            thresh[i], max_value = max(enumerate(histr), key=operator.itemgetter(1))

            #thresh[i] = histr.index(max(histr))
            print thresh[i]

            #plt.plot(histr,color = color[i])
            #plt.xlim([0,256])
        #plt.show()
    if Mode == 1:
        #Creating arrays to store the min and max HSV ranges
        lower = np.array([thresh[0]-30, thresh[1]-30, thresh[2]-40])
        upper = np.array([thresh[0]+30, thresh[1]+30, thresh[2]+40])

        #Create binary mask of in range object
        mask = cv2.inRange(hsv_img, lower, upper)
        #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        erosion = cv2.erode(mask,kernel,iterations = 2)
        dilation = cv2.dilate(erosion,kernel,iterations = 3)
        opening = cv2.dilate(dilation,kernel,iterations = 3)
        dst = cv2.calcBackProject([hsv_img],[0],histrr,[0,180],1)

        ret, track_window = cv2.CamShift(dst, (refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1]), term_crit)
        duplicate[opening == 0] = 0
        
        cv2.ellipse(img, ret, (0, 0, 255), 2)
        
        cv2.imshow('Mask', mask)
        cv2.imshow('Opening', opening)
        cv2.imshow('track', duplicate)
        
        k = cv2.waitKey(1) & 0xFF
        if(k==27):
            break
    cv2.imshow('Camshift', img)
