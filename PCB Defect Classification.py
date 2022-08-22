# PCB Defect Classification code source for *Printed Circuit Board (PCB) 
# Defect Detection and Classification* project. 

#-------
    # Authors: AH. Nazeri, H. Alsalih, CH. Muhta
    # Copyright: "Limited"
    # Version: "2.2.8"
#-------

# Establish Main Folder as Google Drive
"""
from google.colab import drive
drive.mount('/content/drive/')
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as img
from tensorflow.keras import layers, models
from gc import callbacks
import os
import cv2
import tempfile
import zipfile
import os

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity 


class Data_Processor:
    def __init__(self, folder, img_size, des_size, des_box_size):
        self.folder = folder
        self.img_size = img_size
        self.des_size = des_size
        self.des_box_size = des_box_size
        self.num_train = 1000
        self.num_test = 500
        self.num_classes = 6
        self.X_train_loc, self.X_train_temp_loc, self.y_train_loc, \
            self.X_test_loc, self.X_test_temp_loc, self.y_test_loc = self.file_loc()
        self.X_train, self.X_train_temp, self.X_test, self.X_test_temp = self.img_reader()
        self.y_train, self.y_test = self.label_reader()

    def file_loc(self):
        # Reads text files
        with open (self.folder+"/trainval.txt", "r") as f1, open (self.folder+"/test.txt", "r") as f2:
            train_labels = f1.read()
            test_labels = f2.read()

        train_labels, test_labels = train_labels.split("\n"), test_labels.split("\n")
        Temp = np.array([train_labels[i].split(" ") for i in range(self.num_train)])
        X_train_loc_pre, y_train_loc = Temp[:,0], Temp[:,1]
        Temp = np.array([test_labels[i].split(" ") for i in range(self.num_test)])
        X_test_loc_pre, y_test_loc = Temp[:,0], Temp[:,1]

        X_train_loc = []
        X_test_loc = []
        X_train_temp_loc = []
        X_test_temp_loc = []
        for i in range(self.num_train):
            X_train_loc.append(X_train_loc_pre[i].replace(".jpg", "_test.jpg"))
            X_train_temp_loc.append(X_train_loc_pre[i].replace(".jpg", "_temp.jpg"))
        for i in range(self.num_test):
            X_test_loc.append(X_test_loc_pre[i].replace(".jpg", "_test.jpg"))
            X_test_temp_loc.append(X_test_loc_pre[i].replace(".jpg", "_temp.jpg"))

        X_train_loc = np.array(X_train_loc)
        X_train_temp_loc = np.array(X_train_temp_loc)
        X_test_loc = np.array(X_test_loc)
        X_test_temp_loc = np.array(X_test_temp_loc)
        return X_train_loc, X_train_temp_loc, y_train_loc, X_test_loc, X_test_temp_loc, y_test_loc

    def img_reader(self):
        # Read all images and convert to B&W
        X_train = np.zeros([self.num_train, self.des_size, self.des_size], dtype=np.float16)
        X_train_temp = np.zeros([self.num_train, self.des_size, self.des_size], dtype=np.float16)
        X_test = np.zeros([self.num_test, self.des_size, self.des_size], dtype=np.float16)
        X_test_temp = np.zeros([self.num_test, self.des_size, self.des_size], dtype=np.float16)

        for i in range(self.num_train):
            Train = img.imread(self.folder + "/" + self.X_train_loc[i])/255
            T_Temp = img.imread(self.folder + "/" + self.X_train_temp_loc[i])/255

            if len(Train.shape) == 2: #If black and white
                Train = cv2.resize(Train, (self.des_size, self.des_size))
                X_train[i] = Train
            else: #If RGB
                Train = cv2.resize(np.max(Train, axis=-1), (self.des_size, self.des_size))
                X_train[i] = Train

            if len(T_Temp.shape) == 2:
                T_Temp = cv2.resize(T_Temp, (self.des_size, self.des_size))
                X_train_temp[i] = T_Temp
            else:
                T_Temp = cv2.resize(np.max(T_Temp, axis=-1), (self.des_size, self.des_size))
                X_train_temp[i] = T_Temp

        for i in range(self.num_test):
            Test = img.imread(self.folder + "/" + self.X_test_loc[i])/255
            T_Temp = img.imread(self.folder + "/" + self.X_test_temp_loc[i])/255

            if len(Test.shape) == 2:
                Test = cv2.resize(Test, (self.des_size, self.des_size))
                X_test[i] = Test
            else:
                Test = cv2.resize(np.max(Test, axis=-1), (self.des_size, self.des_size))
                X_test[i] = np.max(Test, axis=-1)

            if len(T_Temp.shape) == 2:
                T_Temp = cv2.resize(T_Temp, (self.des_size, self.des_size))
                X_test_temp[i] = T_Temp
            else:
                T_Temp = cv2.resize(np.max(T_Temp, axis=-1), (self.des_size, self.des_size))
                X_test_temp[i] = T_Temp

        return X_train, X_train_temp, X_test, X_test_temp

    def label_reader(self):
        # Read all labels
        y_train = []
        y_test = []
        for i in range(self.num_train):
            with open(self.folder+"/"+self.y_train_loc[i], "r") as f1:
                Temp = f1.read().split("\n")
            Temp = list(filter(None,Temp))
            Temp = np.array([Temp[j].split(" ") for j in range(len(Temp))])
            x1, y1, x2, y2, cs = Temp[:,0], Temp[:,1], Temp[:,2], Temp[:,3], Temp[:,4]

            Temp = np.zeros((self.img_size, self.img_size, 1))
            for j in range(x1.shape[0]):
                Temp = cv2.rectangle(Temp, (int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])), color = int(cs[j]), thickness = int(-1))
            Temp = cv2.resize(Temp, (self.des_box_size, self.des_box_size))
            y_train.append(Temp)
        y_train = np.array(y_train)
                            
        for i in range(self.num_test):
            with open(self.folder+"/"+self.y_test_loc[i], "r") as f1:
                Temp = f1.read().split("\n")
            Temp = list(filter(None,Temp))
            Temp = np.array([Temp[j].split(" ") for j in range(len(Temp))])
            x1, y1, x2, y2, cs = Temp[:,0], Temp[:,1], Temp[:,2], Temp[:,3], Temp[:,4]

            Temp = np.zeros((self.img_size, self.img_size, 1))
            for j in range(x1.shape[0]):
                Temp = cv2.rectangle(Temp, (int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])), color = int(cs[j]), thickness = int(-1))
            Temp = cv2.resize(Temp,(self.des_box_size, self.des_box_size))
            y_test.append(Temp)
        y_test = np.array(y_test)
        return y_train, y_test



# Load Data
folder = "PCBData"
reader = Data_Processor(folder, 640, 128, 88)


plt.imshow(np.reshape(reader.y_train[1],(reader.des_box_size,reader.des_box_size)),cmap='gnuplot')
plt.show()

dataset_train = tf.data.Dataset.from_tensor_slices((reader.X_train, reader.y_train))
dataset_test = tf.data.Dataset.from_tensor_slices((reader.X_test, reader.y_test))

BATCH_NUM = 128

dataset_train = dataset_train.shuffle(reader.num_train)
dataset_test = dataset_test.shuffle(reader.num_test)

dataset_train = dataset_train.batch(BATCH_NUM)
dataset_test = dataset_test.batch(BATCH_NUM)




# Build UNET

epochs = 4
end_step = np.ceil(1.0 * reader.num_train / BATCH_NUM).astype(np.int32) * epochs
print(end_step)

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=10)
}

## layer5 cnn bn he flip
inputs = keras.layers.Input((128,128,1))
# x = keras.layers.experimental.preprocessing.Resizing(128,128)(inputs)
x = keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal_and_vertical')(inputs)

x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=64,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #126 126 64
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)
x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=64,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #124 124 64
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)
x1 = tf.identity(x)
x1 = keras.layers.experimental.preprocessing.CenterCrop(92,92)(x1)

x = keras.layers.MaxPooling2D((2, 2), strides=2)(x) #62 62,64

x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=128,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #60 60 128
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)
x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=128,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #58 58 128
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)
x2 = tf.identity(x)
x2 = keras.layers.experimental.preprocessing.CenterCrop(50,50)(x2)

x = keras.layers.MaxPooling2D((2, 2), strides=2)(x) #29 29 128

x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=256,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #27 27 256
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)
x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=256,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #25 25 256
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=(2, 2),kernel_initializer=tf.keras.initializers.HeNormal())(x) # 50 50 128

x = keras.layers.Concatenate()([x,x2]) #50 50 256
x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=128,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #48 48 128
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)
x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=128,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #46 46 128
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=(2, 2),kernel_initializer=tf.keras.initializers.HeNormal())(x) # 92 92 64

x = keras.layers.Concatenate()([x,x1]) #92,92 128
x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=64,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #90 90 64
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)
x = sparsity.prune_low_magnitude(keras.layers.Conv2D(filters=64,kernel_size=3,kernel_initializer=tf.keras.initializers.HeNormal()), **pruning_params)(x) #88 88 64
x = keras.layers.Activation('relu')(x)
#x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Conv2D(filters=7,kernel_size=1,kernel_initializer=tf.keras.initializers.HeNormal())(x) #88 88 7


model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


import tempfile
from keras.callbacks import ModelCheckpoint

savepath='Checkpoints/5_3_700'
checkpoint = ModelCheckpoint(savepath,monitor='val_loss',mode='min',save_best_only=True,verbose=1)

logdir = tempfile.mkdtemp()
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0),
    checkpoint
]

history = model.fit(reader.X_train, reader.y_train, epochs=700, callbacks=callbacks, validation_data=(reader.X_test, reader.y_test), verbose=1)
# history = model.fit(reader.X_train, reader.y_train, epochs=10, validation_data=(reader.X_test, reader.y_test), verbose=1)



# Strip the pruning wrappers from the pruned model before export for serving

final_pruned_model = sparsity.strip_pruning(pruned_model)
final_pruned_model.summary()


plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.ylim(0.145, 0.17)
plt.xlim(0, 225)
plt.xlabel("Epochs", size = 16)
plt.ylabel("Loss", size = 16)
plt.savefig('Loss.pdf', dpi=720, bbox_inches='tight')
plt.savefig('Loss.PNG', dpi=720, bbox_inches='tight')



plt.figure(figsize=(8,6))
plt.plot(history.history['val_loss'])
plt.ylim(0.14, 0.16)
plt.xlim(0, 225)
plt.xlabel("Epochs", size = 16)
plt.ylabel("Val Loss", size = 16)
plt.savefig('ValLoss.pdf', dpi=720, bbox_inches='tight')
plt.savefig('ValLoss.PNG', dpi=720, bbox_inches='tight')



print(min(history.history["val_loss"]))
print(min(history.history["loss"]))