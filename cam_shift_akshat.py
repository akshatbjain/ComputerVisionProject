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

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

def get_roi(event, x, y, flags, param):
    global refPt, track_window

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        #track_window.append(refPt[0][0], refPt[0][1])

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        #track_window.append(refPt[1][0], refPt[1][1])

        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow('Camshift', img)

cv2.namedWindow('Camshift')
cv2.setMouseCallback('Camshift', get_roi)

while(1):
    ret, img = cam.read()
    duplicate = img.copy()
    hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    cv2.imshow('Camshift', img)
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
        #mask = np.zeros((hsv.shape[0], hsv.shape[1]))
        #mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        #hist = cv2.calcHist([hsv], [0], None , [180], [0, 180])
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
        lower = np.array([thresh[0]-20, thresh[1]-20, thresh[2]-50])
        upper = np.array([thresh[0]+20, thresh[1]+20, thresh[2]+50])

        #Create binary mask of in range object
        mask = cv2.inRange(hsv_img, lower, upper)
        dst = cv2.calcBackProject([hsv_img],[0],histrr,[0,180],1)

        ret, track_window = cv2.CamShift(dst, (refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1]), term_crit)
        duplicate[mask == 0] = 0
        print ret
        cv2.ellipse(duplicate, ret, (0, 0, 255), 2)
        #pts = cv2.boxPoints(ret)
        #pts = np.int0(pts)
        #img2 = cv2.polylines(frame,[pts],True, 255,2)
        #cv2.imshow('img2',img2)
        #Bitwise AND operation on orginal image using mask
        #obj = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('Mask', mask)
        cv2.imshow('track', duplicate)

        #cv2.imshow('Object', obj)
        k = cv2.waitKey(1) & 0xFF
        if(k==27):
            break
