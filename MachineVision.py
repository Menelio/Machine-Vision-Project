#needed imports
import SimpleITK as sitk
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.layers import Conv3D, Concatenate, MaxPooling3D, Reshape
from keras.layers import UpSampling3D, Activation, Permute
from IPython.display import clear_output
import cv2
import imutils as imutils
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # machine learning
from tqdm import tqdm # make your loops show a smart progress meter 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "DataSets/BRATS2015_Training/BRATS2015_Training/HGG")

#process data for model 2
#use Glob to recursively get .mha file
brains = glob.glob(path +"/**/*T1c*.mha",recursive=True)#lacation of brain MRIs
tumors = glob.glob(path+"/**/*OT*.mha",recursive=True)#lacation of segmented brain tumor MRIs

seed = 1
imgSize = (224, 224) # size of vgg16 input

#size to resize images to
size=(32,32,32)

brainImgs= []
tumorImgs = []
for file in brains:
    i = io.imread(file, plugin='simpleitk')
    i = (i-i.mean()) / i.std()
    i = trans.resize(i, size, mode='constant')
    brainImgs.append(i)
np.save("x",np.array(brainImgs)[..., np.newaxis].astype('float32'))#
    
for file in tumors:
    i = io.imread(file, plugin='simpleitk')
    i[i == 4] = 1
    i[i != 1] = 0
    i = i.astype('float32')
    i = trans.resize(i, size, mode='constant')
    tumorImgs.append(i)
np.save("y",np.array(tumorImgs)[..., np.newaxis].astype('float32'))

#load x and y numoy arrays and then plot some random images of brains and tumors to make sure they work
x = np.load('x.npy')
print('x: ', x.shape)
y = np.load('y.npy')
print('y:', y.shape)

inputs = Input(shape=x.shape[1:])#input shape

layer1 = Conv3D(16, 3, activation='elu', padding='same')(inputs)
layer2 = MaxPooling3D()(layer1)

layer3 = Conv3D(16, 3, activation='elu', padding='same')(layer2)
layer4 = MaxPooling3D()(layer3)

layer5 = Conv3D(16, 3, activation='elu', padding='same')(layer4)
layer6 = MaxPooling3D()(layer5)

layer7 = Conv3D(16, 3, activation='elu', padding='same')(layer6)
layer8 = MaxPooling3D()(layer7)

layer9 = Conv3D(16, 3, activation='elu', padding='same')(layer8)

layer10 = UpSampling3D()(layer9)
layer11 = Concatenate(axis=4)([layer7, layer10])

layer12 = Conv3D(16, 3, activation='elu', padding='same')(layer11)

layer13 = UpSampling3D()(layer12)
layer14 = Concatenate(axis=4)([layer5, layer13])

layer15 = Conv3D(16, 3, activation='elu', padding='same')(layer14)

layer16 = UpSampling3D()(layer15)
layer17 = Concatenate(axis=4)([layer3, layer16])

layer18 = Conv3D(16, 3, activation='elu', padding='same')(layer17)

layer19 = UpSampling3D()(layer18)
layer20 = Concatenate(axis=4)([layer1, layer19])

layer21 = Conv3D(16, 3, activation='elu', padding='same')(layer20)

outputs=Conv3D(1, 1, activation='sigmoid')(layer20)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#import os # to find file without having to run first code block
path = os.path.join(my_path, "Weights/model2W.h5")
model.load_weights(path)#location of saved weights
#from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.000001), loss='binary_crossentropy',metrics = ['accuracy'])

#fit model
history = model.fit(x, y, validation_split=0.2, epochs=1, batch_size=8)
print("Training Done")

#save weights for later
#model.save_weights('model2W.h5')


import random as r
pred = model.predict(x[:50])
print(pred)

num = int(x.shape[1]/2)
for n in range(3):
    i = int(r.random() * pred.shape[0])
    plt.figure(figsize=(15,10))

    plt.subplot(131)
    plt.title('Input')
    plt.imshow(x[i, num, :, :, 0])

    plt.subplot(132)
    plt.title('real data')
    plt.imshow(y[i, num, :, :, 0])

    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(pred[i, num, :, :, 0])

    plt.show()


print("Flag")