import pickle, cv2, sys
import numpy as np
import np_util as npu
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


labelPairs = np.genfromtxt('signnames.csv', 
    delimiter=',',skip_header=1, dtype=[('class','i8'),('sign','S50')])
n_classes = len(labelPairs)

## Network Arch
drop = 0.25
layers = [
    Convolution2D(32,3,3, border_mode='valid', input_shape=(32,32,1), activation='relu'),
    Convolution2D(32,3,3, border_mode='valid', activation='relu'),
    MaxPooling2D(),
    Dropout(drop),
    
    Convolution2D(64,3,3, border_mode='valid', activation='relu'),
    Convolution2D(64,3,3, border_mode='valid', activation='relu'),
    MaxPooling2D(),
    Dropout(drop),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
]
myM = Sequential(layers)
myM.summary()


menu = 'python train.py [--archonly]'
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg.endswith('help') or arg=='-h' or arg.startswith('--h'):
        print(menu)
    elif arg!='--archonly':
        print('Unknown argument')
        print(menu)
    exit()


## Train
datapath = '../data/traffic-signs-data/'
trainset = datapath+'train.p'
validset = datapath+'valid.p'
with open(trainset, mode='rb') as f:
    train = pickle.load(f)
with open(validset, mode='rb') as f:
    valid = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']

X_train_gray = npu.grayscale(X_train)
X_valid_gray = npu.grayscale(X_valid)

X_norm = npu.normalize(X_train_gray)
X_val_norm = npu.normalize(X_valid_gray)

y_hot = np_utils.to_categorical(y_train, n_classes)
y_val_hot = np_utils.to_categorical(y_valid, n_classes)

batch_size = 128
epoch = 30

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True)

datagen.fit(X_norm)
X_train_gen = datagen.flow(X_norm, y_hot, shuffle=True, batch_size=batch_size)

myM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
myM.fit_generator(X_train_gen, samples_per_epoch=len(X_norm), nb_epoch=epoch, 
                  validation_data=(X_val_norm, y_val_hot))
myM.save('model.h5')
