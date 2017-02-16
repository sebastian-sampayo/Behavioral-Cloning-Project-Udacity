import csv
import numpy as np
from keras.optimizers import Adam

# ----------------------------------------------------------------------------
# Shapes
img_shape = [160, 320, 3]
# resized_shape = [80, 160, 3]
resized_shape = [64, 64, 3]

# ----------------------------------------------------------------------------
# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 128

# ----------------------------------------------------------------------------
# Program modes
predict_mode = False
validation_mode = True
save_mode = True

# ----------------------------------------------------------------------------
# File names
model_filename = 'model.h5'
# train_log_file = 'test/train.csv'
# validation_log_file = 'test/validation.csv'
# train_log_file = 'data/train.csv'
# validation_log_file = 'data/validation.csv'
# train_log_file = 'three_img/train.csv'
# train_log_file = 'data/train_random.csv'
# validation_log_file = 'data/validation_random.csv'

# We are using data augmentation, so validation images will be slightly different from training data
# original_log_file = 'data/driving_log.csv'
train_log_file = 'data/driving_log_shuffled.csv'
validation_log_file = 'data/driving_log_shuffled2.csv'

# ----------------------------------------------------------------------------
# Data augmentation
augmentation_factor = 8
validation_factor = 0.2

# Angle correction for side cameras
shift_angle = 0.25

# Translation ranges for data augmentation
x_translation_range = img_shape[1]/20 # /20 = 8 # 8/64=1/8 ## /3.2 = 100
y_translation_range = img_shape[0]/50 # /50 = 6.4 # 6.4/64 = 1/10 ## /4 = 40
translation_shift_angle = 0.15
GAUSSIAN_TRANSLATION = True

# fix random seed for reproducibility
seed = 3
np.random.seed(seed)

# Randomly select the camera (center, left or right)
# non uniform: "center_camera_prob" chance for center,
# (1 - center_camera_prob) for side camera (evenly split into left and right)
center_camera_prob = 0.34
# ----------------------------------------------------------------------------
# CSV configuration
class MyDialect(csv.Dialect):
    strict = True
    skipinitialspace = True
    # quoting = csv.QUOTE_ALL
    quoting = csv.QUOTE_NONE
    delimiter = ','
    # quotechar = "'"
    lineterminator = '\n'
    escapechar='\\'
    
# ----------------------------------------------------------------------------
# Configures the learning process and metrics
# loss = 'sparse_categorical_crossentropy'
# loss = 'categorical_crossentropy'
loss = 'mean_squared_error'
# learning_rate = 0.0005
# optimizer = Adam(lr=learning_rate)
optimizer = 'Adam'

# ----------------------------------------------------------------------------
# Autonomous driving
driving_throttle = 0.3