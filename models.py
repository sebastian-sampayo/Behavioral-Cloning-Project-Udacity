import math
from keras.models import Sequential, Model
from keras.layers import Input, GlobalAveragePooling2D, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Merge, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from params import *
# =============================================================================================== #
# Build a Multi-layer feedforward neural network with Keras here.

### Images preprocessing: resize and normalization.
def normalize_image(image):
  return image / 127.5 - 1

def resize(image):
  import tensorflow as tf
  return tf.image.resize_images(image, [64, 64])
  # return tf.image.resize_images(image, resized_shape[:2])
    
def nvidia_like_model():
  model = Sequential()
  # Crop
  crop_bottom = math.floor(img_shape[0]/6)
  crop_top = crop_bottom * 2
  model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=img_shape, name='input'))
  # Resize to 64x64
  model.add(Lambda(resize))
  # Normalize to (-1, 1)
  model.add(Lambda(normalize_image))

  # In: 64x64
  model.add(Convolution2D(3, 1, 1, border_mode='same', activation='elu'))

  model.add(Convolution2D(3, 5, 5, border_mode='same', activation='elu'))
  model.add(MaxPooling2D((2, 2)))

  # In: 32x32
  model.add(Convolution2D(24, 5, 5, border_mode='same',activation='elu'))
  model.add(MaxPooling2D((2, 2)))

  # In: 16x16
  model.add(Convolution2D(36, 5, 5, border_mode='same',activation='elu'))
  model.add(MaxPooling2D((2, 2)))

  # In: 8x8
  model.add(Convolution2D(48, 5, 5, border_mode='same',activation='elu'))
  model.add(MaxPooling2D((2, 2)))

  # In: 4x4
  model.add(Convolution2D(64, 3, 3, border_mode='same',activation='elu'))
  model.add(MaxPooling2D((2, 2)))

  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(1024, activation='elu'))
  model.add(Dense(100,activation='elu' ))
  model.add(Dense(50, activation='elu'))
  model.add(Dense(10, activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(1,name='output'))
  
  return model

  
def nvidia_like_model_2():
  model = Sequential()
  # Crop
  crop_bottom = math.floor(img_shape[0]/6)
  crop_top = crop_bottom * 2
  model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=img_shape, name='input'))
  # Resize to 64x64
  model.add(Lambda(resize))
  # Normalize to (-1, 1)
  model.add(Lambda(normalize_image))

  # In: 64x64
  model.add(Convolution2D(3, 1, 1, border_mode='same', activation='elu'))
  # model.add(Dropout(0.5))

  model.add(Convolution2D(3, 5, 5, border_mode='same', activation='elu'))
  model.add(Convolution2D(3, 5, 5, border_mode='same', activation='elu'))
  model.add(MaxPooling2D((2, 2)))
  # model.add(Dropout(0.5))

  # In: 32x32
  model.add(Convolution2D(24, 5, 5, border_mode='same',activation='elu'))
  model.add(Convolution2D(24, 5, 5, border_mode='same',activation='elu'))
  model.add(MaxPooling2D((2, 2)))

  # In: 16x16
  model.add(Convolution2D(36, 5, 5, border_mode='same',activation='elu'))
  model.add(Convolution2D(36, 5, 5, border_mode='same',activation='elu'))
  model.add(MaxPooling2D((2, 2)))
  # model.add(Dropout(0.5))

  # In: 8x8
  model.add(Convolution2D(48, 5, 5, border_mode='same',activation='elu'))
  model.add(Convolution2D(48, 5, 5, border_mode='same',activation='elu'))
  model.add(MaxPooling2D((2, 2)))

  # In: 4x4
  model.add(Convolution2D(64, 3, 3, border_mode='same',activation='elu'))
  model.add(Convolution2D(64, 3, 3, border_mode='same',activation='elu'))
  model.add(MaxPooling2D((2, 2)))
  
  # In: 2x2
  model.add(Convolution2D(128, 3, 3, border_mode='same',activation='elu'))
  model.add(Convolution2D(128, 3, 3, border_mode='same',activation='elu'))

  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(1024, activation='elu'))
  model.add(Dense(100,activation='elu' ))
  model.add(Dense(50, activation='elu'))
  model.add(Dense(10, activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(1,name='output'))
  
  return model

# Try: 96x96 at the input, and adding conv 128x3x3 with max pooling
# or: just adding conv 128x3x3 without maxpooling !

# Loads model filename ".h5" file and make trainable only the last layer for fine-tunning
def loaded_model(filename='model.h5'):
  print()
  print('Loading model...')
  model = load_model(filename)
  # Set every layer to be not trainable
  for layer in model.layers:
    layer.trainable = False
  # Set to trainable only the last layer for fine-tunning
  model.layers[-1].trainable = True
  print('Model loaded.')
  print()
  return model