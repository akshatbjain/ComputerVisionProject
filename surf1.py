import cv2
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
img=cv2.imread('w1n.jpg',0)
img1=cv2.imread('w2n.jpg',0)
img2=cv2.imread('w3n.jpg',0)
img3=cv2.imread('w4n.jpg',0)
des0=np.zeros((1,32))
surf = cv2.ORB_create()
kp, des = surf.detectAndCompute(img,None)
kp1, des1 = surf.detectAndCompute(img1,None)
#ret1,frame1=vid.read()
# print len(kp),des
#print 'len',des.shape
#k=int(len(des)*0.25)
# print 'k',k
bf = cv2.BFMatcher()
matches = bf.match(des,des1)
matches = sorted(matches, key = lambda x:x.distance)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
img3 = cv2.drawMatches(img,kp,img1,kp1,matches[:400],None,flags=2)
plt.imshow(img3)
plt.show()

exit()
des_val=np.zeros((1,32))

clf = svm.SVC(kernel='poly',degree=5,probability=True)
kp, des = surf.detectAndCompute(img,None)
k=int(len(des)*0.25);
#print 'k',k
k_index=random.sample(range(0,k),k);
#print 'k_in',len(k_index)
des_val=np.vstack((des_val,des[k_index]))
#print 'des val',des_val.shape
des=np.delete(des,(k_index),axis=0)
#print 'des',des.shape
des0=np.vstack((des0,des))
#print 'des',des0.shape
kp,des=surf.detectAndCompute(img1,None)
k=int(len(des)*0.25);
k_index=random.sample(range(0,k),k);
des_val=np.vstack((des_val,des[k_index]))
des=np.delete(des,(k_index),axis=0)
des0=np.vstack((des0,des))
kp,des=surf.detectAndCompute(img2,None)
k=int(len(des)*0.25);
k_index=random.sample(range(0,k),k);
des_val=np.vstack((des_val,des[k_index]))
des=np.delete(des,(k_index),axis=0)
des0=np.vstack((des0,des))
kp,des=surf.detectAndCompute(img3,None)
k=int(len(des)*0.25);
k_index=random.sample(range(0,k),k);
des_val=np.vstack((des_val,des[k_index]))
des=np.delete(des,(k_index),axis=0)
des0=np.vstack((des0,des))

vid=cv2.VideoCapture(0)
count=0
new_des=np.zeros((1,32))
while(vid.isOpened() and count<20):        
    ret1,frame1=vid.read()
    ngray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    kp2, des2 = surf.detectAndCompute(ngray,None)
    new_des=np.vstack((new_des,des2))
    count=count+1
vid.release()
des0 = np.delete(des0, (0), axis=0)
des_val= np.delete(des0, (0), axis=0)
new_des = np.delete(new_des, (0), axis=0)
one=np.ones((len(des0),1))
zer=np.zeros((len(new_des),1))
o=np.vstack((one,zer))
des0=np.vstack((des0,new_des))
print des0.shape
print des0
print o
X_train, X_test, y_train, y_test = train_test_split(des0, o, test_size=0.20)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print 'accuracy',accuracy_score(y_test,y_pred)

cap=cv2.VideoCapture(0)

thresh = 0.80 

    
k=KMeans(n_clusters=3)  

cnt_1=np.array([0,0,0])
ind=0

while(cap.isOpened()):      
    a1=[]
    b1=[]
    indxkm = []
    cnt=[]
    ret, frame3 = cap.read()    
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    


    kp3, des3 = surf.detectAndCompute(gray3,None) 
    try:

        out=clf.predict_proba(des3)
        #print 'out',out 
    except Exception:
        pass
    cnt.append(1)
    for i in range(len(out)):

        if out[i][1] >= thresh:   
            cnt=np.append(cnt,i)
    if len(cnt)>10:     
        for mat in range(len(cnt)):
            index = cnt[mat]
            (x1,y1) = kp3[index].pt
            a1.append(x1)
            b1.append(y1)
            x=np.c_[a1,b1]


        k.fit(x)      
        cent = k.cluster_centers_


        for i in range(len(k.labels_)): 
            cnt_1[k.labels_[i]]=cnt_1[k.labels_[i]]+1  
        lab=np.max(cnt_1)
        for i in range(len(cnt_1)):
            if lab==cnt[i]:
                ind=i


        cv2.circle(frame3, (int(cent[ind][0]),int(cent[ind][1])), 20, (0, 255,0), 5)  
    else:
        pass
       
    cv2.imshow('detect',frame3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
