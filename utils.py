# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 3: Behavioral Cloning
# Date: 12th February 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: utils.py
# =========================================================================== #
# Utility functions

import cv2
import math
import numpy as np
import csv
import io
import matplotlib.image as mpimg
from params import *
from sklearn.utils import shuffle

# ----------------------------------------------------------------------------
# Get the number of rows in a csv:
def count_rows(file):
  with open(file, 'rt') as csvfile:
      rows = csv.reader(csvfile, delimiter=',')
      n_rows = 0
      for row in rows:
          n_rows += 1
  return n_rows

# ----------------------------------------------------------------------------
# Parse CSV line
# return list with line elements
def parse_csv_line(line):
  line_string = io.StringIO(line)
  line_csv = csv.reader(line_string, MyDialect()) # skip initial space
  output = []
  for row in line_csv: # only one row
    for item in row:
      output.append(item)
  return output

# ----------------------------------------------------------------------------
# Shuffle driving log csv file  
def shuffle_csv(input_file, output_file):
  
  # Read CSV input file, shuffle lines
  f = open(input_file, 'rt')
  with f as csvfile:
    reader = csv.reader(csvfile, MyDialect())
    driving_data = [row for row in reader]
    train_data = shuffle(driving_data)
  f.close()
    
  # Write CSV output file
  f = open(output_file, 'wt')
  with f as csvfile:
    spamwriter = csv.writer(csvfile, MyDialect())
    for row in train_data:
      spamwriter.writerow(row)
  f.close()

# ----------------------------------------------------------------------------
# translate an image randomly within a certain range and adjust the steering angle accordingly
# ecuations from:
# https://carnd-forums.udacity.com/questions/36059111/answers/36059810
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.4kpw1hort
def translate_image(image, angle, x_range, y_range):
    # Translation
    # x_shift = x_range * np.random.uniform() - x_range / 2
    x_shift = np.random.normal(0, x_range/2)
    shifted_angle = angle + x_shift / x_range * 2 * translation_shift_angle
    y_shift = y_range * np.random.uniform() - y_range / 2

    modifier = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    size = (img_shape[1], img_shape[0])
    translated_image = cv2.warpAffine(image, modifier, size)

    return translated_image, shifted_angle

# ----------------------------------------------------------------------------
# Randomly flip an image and invert steering angle (50% chance)
# to simulate left turns from right turns and vice-versa
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1zm9bk6xi
def flip_image(image, angle):
  flipped_angle = angle
  ind_flip = np.random.randint(2)
  if ind_flip==0:
      image = cv2.flip(image,1)
      flipped_angle = -angle
  return image, flipped_angle

# ----------------------------------------------------------------------------
# Randomply augment brightness to simulate day and night conditions
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1zm9bk6xi
def augment_brightness_camera_images(image):
  image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
  random_bright = .25+np.random.uniform()
  image1[:,:,2] = image1[:,:,2]*random_bright
  image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
  return image1
    
# ----------------------------------------------------------------------------
# Randomly add shadows across the image. This is implemented by choosing random points and shading all points on one side (chosen randomly) of the image. 
# Do it for 1 image out of 4
def add_random_shadow(image):
  top_y = img_shape[0]*np.random.uniform()
  top_x = 0
  bot_x = img_shape[1]
  bot_y = img_shape[0]*np.random.uniform()
  image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
  shadow_mask = 0*image_hls[:,:,1]
  X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
  Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
  shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
  if np.random.randint(4)==1:
    random_bright = .25+.7*np.random.uniform()
    # random_bright = .5
    cond1 = shadow_mask==1
    cond0 = shadow_mask==0
    if np.random.randint(2)==1:
      image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
    else:
      image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
  image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
  return image
    
# ----------------------------------------------------------------------------
# Transform image (translation, flip, rotation, etc)
def data_augmentation(img, steering_angle):
  img, steering_angle = translate_image(img, steering_angle, x_translation_range, y_translation_range)
  # img = add_random_shadow(img) # just for robustness
  img = augment_brightness_camera_images(img) # just for robustness
  img, steering_angle = flip_image(img, steering_angle)
  return img, steering_angle

  
# ----------------------------------------------------------------------------
# Parse the csv row, and load an image from one of the three cameras available
# Adjust the steering angle accordingly
def process_line(line):
    output = parse_csv_line(line)
    
    center_idx = 0
    left_idx = 1
    right_idx = 2
    steering_angle_idx = 3
    # Randomly select the camera (center, left or right)
    # non uniform: 40% chance for center, 30% left, 30% right
    dice = np.random.randint(100)
    diceLR = np.random.randint(2) + 1 # 1, 2
    if dice > center_camera_prob*100:
      x_idx = diceLR
    else:
      x_idx = center_idx
    # x_idx = right_idx # debug
    y_idx = steering_angle_idx
    
    img = mpimg.imread(output[x_idx])
    # Convert string to float
    steering_angle = float(output[y_idx])
    # Adjust the steering angle according to the camera selected
    if x_idx == left_idx:
      steering_angle = steering_angle + shift_angle
    if x_idx == right_idx:
      steering_angle = steering_angle - shift_angle
    return img, steering_angle

# ----------------------------------------------------------------------------
def generate_arrays_from_file(path, batch_size, predict=False):
  # j = 0
  while 1:
    # j += 1
    # Shuffle data
    shuffle_csv(path, path)
    # print()
    # print('Train data loop: ', j)
    
    f = open(path)
    i = 0
    x = []
    y = []
    for line in f:
      # create numpy arrays of input data
      # and labels, from each line in the file
      img, steering_angle = process_line(line)
      img, steering_angle = data_augmentation(img, steering_angle)
      x.append(img)
      y.append(steering_angle)
      i += 1
      if i == batch_size:
        x = np.asarray(x)
        y = np.asarray(y)
        if (predict == False):
          yield (x, y)
        else:
          yield (x)
        i = 0
        x = []
        y = []
    f.close()
