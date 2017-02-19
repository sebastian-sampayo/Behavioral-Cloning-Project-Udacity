import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as sci

from utils import *
from params import *
from models import *

# ----------------------------------------------------------------------------
# Get the number of samples in the csv
n_train_samples = count_rows(train_log_file)
print('Number of training samples: ', n_train_samples)

# Augment number of training samples by increasing samples per epoch  
# n_train_samples *= augmentation_factor
# n_train_samples *= 3
n_train_samples = 1
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
# filename = 'analysis/gauss_translation_LR_{}_{}_{:.2f}_{:.2f}.png'.format(int(center_camera_prob*100), n_train_samples, shift_angle, translation_shift_angle)
# title = 'Gaussian translation'
img_dir = 'analysis/'
title_before = 'Original image'
title_crop = 'Cropped image'
title_resized = 'Resized image'
title_norm = 'Normalized image'
crop_bottom = math.floor(img_shape[0]/6)
crop_top = crop_bottom * 2
def crop_image(image):
  import math
  # crop out the top 1/6 with horizon line & bottom containing hood of the car
  image_shape = image.shape
  y_offset =  math.floor(image_shape[0]/6)
  image = image[crop_top:(image_shape[0] - crop_bottom), 0:image_shape[1]]
  return image
# ----------------------------------------------------------------------------
i = 0
angles = []
for out in generate_arrays_from_file(train_log_file, 1):
  angle = out[1]
  angles.append(angle)
  img = out[0][0]
  
  img_trans, angle_trans = translate_image(img, angle, x_translation_range, y_translation_range)
  # img = add_random_shadow(img) # just for robustness
  img_bright = augment_brightness_camera_images(img_trans) # just for robustness
  angle_bright = angle_trans
  img_flip, angle_flip = flip_image(img_bright, angle_bright)
  
  img_crop = crop_image(img_flip)
  img_resized = sci.imresize(img_crop, (64, 64))
  
  plt.imshow(img)
  plt.title('Original. Steering angle = {}'.format(angle))
  plt.savefig(img_dir + 'original_img2.png')
  plt.show()
  
  plt.imshow(img_trans)
  plt.title('Translated image. Steering angle = {}'.format(angle_trans))
  plt.savefig(img_dir + 'translated.png')
  plt.show()
  
  plt.imshow(img_bright)
  plt.title('Brightness augmentated image. Steering angle = {}'.format(angle_bright))
  plt.savefig(img_dir + 'brightness.png')
  plt.show()
  
  plt.imshow(img_flip)
  plt.title('Flipped image. Steering angle = {}'.format(angle_flip))
  plt.savefig(img_dir + 'flip.png')
  plt.show()
  
  plt.imshow(img_resized)
  plt.title('Cropped and resized image. Steering angle = {}'.format(angle_flip))
  plt.savefig(img_dir + 'augment_resized.png')
  plt.show()
  
  i += 1
  if i%1000 == 0:
    print('i: ', i)
  if i==n_train_samples:
    break

# angles = np.asarray(angles)
# h = plt.hist(angles, bins=100)
# plt.xlim([-1, 1])
# plt.ylabel('Amount of samples')
# plt.xlabel('Steering angle')
# plt.title(title)
# plt.savefig(filename)
# plt.show()
