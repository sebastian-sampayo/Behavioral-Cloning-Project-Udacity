import matplotlib.pyplot as plt
import numpy as np

from utils import *
from params import *

# ----------------------------------------------------------------------------
# Get the number of samples in the csv
n_train_samples = count_rows(train_log_file)
print('Number of training samples: ', n_train_samples)

# Augment number of training samples by increasing samples per epoch  
# n_train_samples *= augmentation_factor
# n_train_samples *= 3
# n_train_samples = 10
BATCH_SIZE = 1
print('New number of training samples: ', n_train_samples)


# ----------------------------------------------------------------------------
# filename = 'analysis/original_data.png'
# title = 'Original data histogram for center camera only'
# filename = 'analysis/translation.png'
# title = 'Using random shifts'
# filename = 'analysis/translation_LR_{}_{}_{:.2f}_{:.2f}.png'.format(int(center_camera_prob*100), n_train_samples, shift_angle, translation_shift_angle)
# title = 'Using shifts and side cameras. Center camera probability: {}%'.format(int(center_camera_prob*100))
# filename = 'analysis/LR_{}_{}.png'.format(int(center_camera_prob*100), n_train_samples)
# title = 'Using side cameras. Center camera probability: {}%'.format(int(center_camera_prob*100))
# filename = 'analysis/translation_R.png'
# title = 'Right camera + random shifts'
filename = 'analysis/gauss_translation_LR_{}_{}_{:.2f}_{:.2f}.png'.format(int(center_camera_prob*100), n_train_samples, shift_angle, translation_shift_angle)
title = 'Gaussian translation'
# ----------------------------------------------------------------------------
i = 0
angles = []
for out in generate_arrays_from_file(train_log_file, 1):
  angle = out[1]
  angles.append(angle)
  img = out[0][0]
  # plt.imshow(img)
  # plt.show()
  i += 1
  if i%1000 == 0:
    print('i: ', i)
  if i==n_train_samples:
    break

angles = np.asarray(angles)
h = plt.hist(angles, bins=100)
plt.xlim([-1, 1])
plt.ylabel('Amount of samples')
plt.xlabel('Steering angle')
plt.title(title)
plt.savefig(filename)
plt.show()
