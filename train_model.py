import numpy as np
import math
from keras.models import Sequential, Model

# User functions, models and configuration parameters
from params import *
from utils import *
from models import *

# ----------------------------------------------------------------------------
# Build a Multi-layer feedforward neural network with Keras here.

# model = nvidia_like_model_2()
model = loaded_model()

# ----------------------------------------------------------------------------
# Get the number of samples in the csv
n_train_samples = count_rows(train_log_file)
# n_train_samples = BATCH_SIZE*20 # DEBUG
print('Number of training samples: ', n_train_samples)
if validation_mode == True:
  n_validation_samples = count_rows(validation_log_file)
  # Make it multiple of BATCH_SIZE
  n_validation_samples -= (n_validation_samples % BATCH_SIZE)
  # n_validation_samples = n_train_samples # DEBUG
  print('Number of validation samples: ', n_validation_samples)
  
# Augment number of training samples by increasing samples per epoch  
n_train_samples *= augmentation_factor
# Make it multiple of BATCH_SIZE
n_train_samples -= (n_train_samples % BATCH_SIZE)

n_validation_samples = math.floor(n_train_samples * validation_factor)
# n_train_samples = 64 # DEBUG
# n_validation_samples = 64 # DEBUG
print('New number of training samples: ', n_train_samples)
print('New number of validation samples: ', n_validation_samples)

# ----------------------------------------------------------------------------
# Configures the learning process and metrics
model.compile(optimizer=optimizer, loss=loss)

# Train the model
if (validation_mode == True):
  print('With validation')
  model.fit_generator(generate_arrays_from_file(train_log_file, BATCH_SIZE),
        samples_per_epoch=n_train_samples, nb_epoch=EPOCHS, 
        validation_data=generate_arrays_from_file(validation_log_file, BATCH_SIZE),
        nb_val_samples=n_validation_samples)
else:
  print('No validation')
  model.fit_generator(generate_arrays_from_file(train_log_file, BATCH_SIZE),
        samples_per_epoch=n_train_samples, nb_epoch=EPOCHS)
  
# ----------------------------------------------------------------------------
if (predict_mode == True):
    print('Predicting...')
    # image_array = np.asarray(image)
    # steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
    predictions = model.predict_generator(generate_arrays_from_file(train_log_file, BATCH_SIZE, predict=True), val_samples=n_train_samples)
    print('Done')
    steering_angle = (predictions)
    i = 0
    y = []
    for item in generate_arrays_from_file(train_log_file, BATCH_SIZE):
      j = 0
      for output in item:
        if j == 0:
          j += 1
          continue
        y.append(output)
        j += 1
      i += 1
      if i == 3:
        break
    print('Original steering angle: ', np.asarray(y))
    print('Predicted steering angle: ', steering_angle)
    
# ----------------------------------------------------------------------------
# Save the model
if (save_mode == True ):
    print('Saving model...')
    model_filename = 'model.h5'
    model.save(model_filename)  # creates a HDF5 file 'model.h5'
    
    # POST PROCESSING, SAVE MODEL TO DISK
    # with open('model.json', 'w') as json_file:
      # json_file.write(model.to_json())

    # save weights as model.h5
    # model.save_weights('model_weights.h5')

    print('Model saved.')
    
# Note on saving model:
# When saving the model I got the following error:
# File: Miniconda3\envs\IntroToTensorFlow\lib\site-packages\keras\utils\generic_utils.py", line 71, in func_dump, line 71, in func_dump
# UnicodeDecodeError: 'rawunicodeescape' codec can't decode bytes in position ..-..: truncated \uXXXX
# a quick patch i am using in generic_utils.py/func_dump(func) is:
# code = marshal.dumps(func.__code__).replace(b'\\',b'/').decode('raw_unicode_escape')