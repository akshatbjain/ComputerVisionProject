import numpy as np
import cv2
from matplotlib import pyplot as plt

img=cv2.imread('w1.jpg',0)
print img.shape
#img=cv2.resize(img,dsize=(420,418),interpolation = cv2.INTER_CUBIC)
#cap=cv2.VideoCapture('wvid.mp4')
img1=cv2.imread('wi1.jpg',0)
print img1.shape
# while(cap.isOpened()):
	# ret,img1=cap.read()

surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(img,None)
kp2, des2 = surf.detectAndCompute(img1,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
 	if m.distance < 0.7*n.distance:
 		matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = 0)

faces=cv2.drawMatchesKnn(img,kp1,img1,kp2,matches,None,**draw_params)
# print faces.shape
# for (x,y,w,0) in faces:
	
# 	cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(faces)
plt.show()
# if cv2.waitKey(1) & 0xFF == ord('q'):
# 	break
# cap.release()	
# cv2.destroyAllWindows()