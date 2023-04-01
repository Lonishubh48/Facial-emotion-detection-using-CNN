#!/usr/bin/env python
# coding: utf-8

# # Facial Emotion Recognition using CNN

# # Import libraries 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import Model
from keras.utils import plot_model

import os
print(os.listdir("E:/study materials/M22 SEM_2/Neural_networt/mini project/Fer_2013"))


# In[3]:


pwd


# # Dataset Overview

# In[4]:


#loading the path of training and testing data directory 
train_data_dir = pathlib.Path("E:/study materials/M22 SEM_2/Neural_networt/mini project/Fer_2013/training")
print(train_data_dir)

test_data_dir = pathlib.Path("E:/study materials/M22 SEM_2/Neural_networt/mini project/Fer_2013/testing")
print(test_data_dir)


# In[5]:


# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory('E:/study materials/M22 SEM_2/Neural_networt/mini project/Fer_2013/training',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'E:/study materials/M22 SEM_2/Neural_networt/mini project/Fer_2013/testing',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


# # Now Build convolutional neural network

# **CNN Architecture:**
# 
#     Conv -> Activation  
#     Conv -> Activation 
#     MaxPooling
#     Dropout
#     
#     Conv -> Activation
#     MaxPooling
#     Conv -> Activation
#     Dropout
#     
#     Flatten
#     Dense ->  Activation
#     Dropout
#     Dense ->  Activation
#     Dense ->  Activation
#     Output layer

# # create model structure

# In[6]:


# create model structure
emotion_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.25),
    Dense(512, activation='relu'),
    Dense(7, activation='softmax')
])

emotion_model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

emotion_model.summary()

# Train the neural network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=7178 // 64)


# In[7]:


import cv2
# retrieve weights from the first hidden layer
filters = emotion_model.layers[0].get_weights()[0]


plt.figure(figsize=(4,4))
for i in range(1):
    plt.subplot(1,1,i+1)
    #showing the result of weights of first kernel 
    plt.imshow(filters[:, :, 0, i], cmap='gray')
    plt.axis('off')
    plt.title("Resut_of weights of first kernel ")
    


# In[33]:


from keras.models import Model
first_layer_model = Model(inputs=emotion_model.inputs, outputs=emotion_model.layers[0].output)

# visualize the feature maps for the first image in the test dataset
test_image = cv2.imread(r'E:\study materials\M22 SEM_2\Neural_networt\mini project\Fer_2013\testing\disgust\PrivateTest_84635755.jpg', 0)
test_image = cv2.resize(test_image, (48, 48))
plt.subplot(1,2,1)
plt.imshow(test_image)
plt.title('Original image')
test_image = np.expand_dims(test_image, axis=0)
test_image = np.expand_dims(test_image, axis=-1)
features = first_layer_model.predict(test_image)

# plot the feature maps
plt.figure(figsize=(4,4))
for i in range(1):
    plt.subplot(1, 1, i+1)
    plt.imshow(features[0, :, :, i], cmap='gray')
    plt.axis('off')
plt.title('Result of first conv_layer') 
plt.show()
plt.savefig('convolved.jpg')


# # Visualize Training Performance

# In[13]:


print(emotion_model_info.history.keys())


# In[15]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,2, figsize=(18, 6))
# Plot training & validation accuracy values
axes[0].plot(emotion_model_info.history['accuracy'])
axes[0].plot(emotion_model_info.history['val_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
axes[1].plot(emotion_model_info.history['loss'])
axes[1].plot(emotion_model_info.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[16]:


emotion_model.save('facial_emotions_detection_model.h5')


# # Test/Predict Image

# In[27]:


from tensorflow.keras.preprocessing import image
img_path=r"E:\study materials\M22 SEM_2\Neural_networt\mini project\Fer_2013\testing\disgust\PrivateTest_84635755.jpg"
test_image=image.load_img(img_path,target_size=(224,224),color_mode='rgb')
plt.subplot(1,2,1)
plt.imshow(test_image)
plt.title('Original_Image')
test_image=image.load_img(img_path,target_size=(48,48),color_mode='grayscale')
test_image=image.img_to_array(test_image)
print(test_image.shape)
plt.subplot(1,2,2)
plt.imshow(test_image, cmap = 'gray')
plt.title('Grayscale_Image')
plt.show()


# In[28]:


test_image=test_image.reshape(1,48,48,1)
classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
result=emotion_model.predict(test_image)
print(result[0])
y_pred=np.argmax(result[0])
print('The person facial emotion is:',classes[y_pred])


# In[ ]:




