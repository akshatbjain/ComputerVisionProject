from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

batch_size = 20
nClasses = 2
dataAugmentation = False

#################################################
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/home/rohit/cv/Training_Dataset/',
        target_size=(224, 224),
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        '/home/rohit/cv/Validation_Dataset/',
        target_size=(224, 224),
        batch_size=batch_size)
#################################################

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nClasses))

#model.load_weights("vgg16_model_11-20_epochs.h5")

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=2,
        validation_data=validation_generator,
        nb_val_samples=400,verbose = 1)

file = open("training_history.txt", "w")
file.write(str(history.history['acc']))
file.write(',')
file.write(str(history.history['val_acc']))
file.write(',')
file.write(str(history.history['loss']))
file.write(',')
file.write(str(history.history['val_loss']))
file.write(',')
file.close()

# serialize model to JSON
model_json = model.to_json()
with open("VGG16_Model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("VGG16_Model.h5")
print("Saved model to disk")
