from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np
import os
import cv2

# load json and create model
json_file = open('VGG16_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("VGG16_Model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])

#loaded_model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

# Code to load the image from webcam
cam = cv2.VideoCapture(0)

print 'Press <q> to exit this code!'

while(1):
    ret, img = cam.read()

    # Code to load the image from local directory
    #im = cv2.imread('test_images/5.jpeg')
    im = img.copy()
    im = cv2.resize(im, (224, 224)).astype(np.float32)
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    out = loaded_model.predict(im)
    print out[0]

    if out[0][0] > 0.8 and out[0][1] > 0.022:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Object Detected', (20,150), font, 2, (255,255,255), 2, cv2.CV_AA)

    cv2.imshow('Input', img)

    k = cv2.waitKey(33)

    if k == 1048689:
	cam.release()
	break
