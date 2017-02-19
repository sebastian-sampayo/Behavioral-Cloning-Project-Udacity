import csv
import numpy as np
from keras.optimizers import Adam

# ----------------------------------------------------------------------------
# Shapes
img_shape = [160, 320, 3]
resized_shape = [64, 64, 3]

# ----------------------------------------------------------------------------
# Program modes
predict_mode = False
validation_mode = True
save_mode = True

# ----------------------------------------------------------------------------
# File names
model_filename = 'model.h5'

train_log_file = 'data/curves_data2/driving_log.csv'
validation_log_file = 'data/validation/driving_log.csv'

# ----------------------------------------------------------------------------
# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 128

# ----------------------------------------------------------------------------
# Data augmentation
augmentation_factor = 4
validation_factor = 0.2

# Angle correction for side cameras
shift_angle = 0.25

# Allow random translation for data augmentation
TRANSLATION = False
# Translation ranges for data augmentation
x_translation_range = img_shape[1]/20 # /20 = 8 ## /3.2 = 100
y_translation_range = img_shape[0]/50 # /50 = 6.4 ## /4 = 40

translation_shift_angle = 0.15
GAUSSIAN_TRANSLATION = True

# fix random seed for reproducibility
seed = 3
np.random.seed(seed)

# Randomly select the camera (center, left or right)
# non uniform: "center_camera_prob" chance for center,
# (1 - center_camera_prob) for side camera (evenly split into left and right)
center_camera_prob = 0.7
# If angle is straight, discard 2 out of 3 times
DISCARD_STRAIGHT = True
# ----------------------------------------------------------------------------
# CSV configuration
class MyDialect(csv.Dialect):
    strict = True
    skipinitialspace = True
    quoting = csv.QUOTE_NONE
    delimiter = ','
    lineterminator = '\n'
    escapechar='\\'
    
# ----------------------------------------------------------------------------
# Configures the learning process and metrics
loss = 'mean_squared_error'
optimizer = 'Adam'

# ----------------------------------------------------------------------------
# Autonomous driving
driving_throttle = 0.3