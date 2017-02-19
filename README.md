TODO:
- Appropriate training data
- References
- code line quotes

# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

** Resume
This project was amazing. I learned a lot about learning models

[//]: # (Image References)

[preprocessing]: ./analysis/preprocessing.png
[model]: ./analysis/model.png
[original_data]: ./analysis/original_data.png
[translation]: ./analysis/translation.png
[addition]: ./analysis/addition.png
[LR0315]: ./analysis/translation_LR_34_64288_0.30_0.15.png
[LR0250125]: ./analysis/translation_LR_34_64288_0.25_0.12.png
[gauss025015]: ./analysis/gauss_translation_LR_34_64288_0.25_0.15.png
[side_cameras]: ./analysis/LR_34_8036.png
[generated_img_C]: ./analysis/augmented_C.png
[generated_img_R]: ./analysis/augmented_R.png

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train_model.py containing the script to create and train the model
* models.py containing a couple of models in Keras to choose for in train_model.py
* utils.py containing several utilitarian functions used all over the project
* params.py with a number of parameters to configure the project (epochs, batch_size, data augmentation parameters, etc)
[//]: # (* analyse_data.py for analysis of the training data.)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The train_model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture

####1. An appropriate model architecture has been employed

[//]: # (My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) )
[//]: # (The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). )

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 3 and 128 (models.py)
The model includes ELU layers to introduce nonlinearity and make transition between angles smoother, as the activation function is continuous (in contrast with RELU activation). 
Besides, the input image is cropped removing 1/6 of the image height (26 pixels) from the bottom and 1/3 (52 pixels) from the top. 
This way we focus on the road, without paying attention to anything else in the background and zoom in the part of the image that contains the curve information.
Furthermore, I resize this cropped image to 64x64, in order to reduce memory usage, and then normalize the values between -1 and 1, using Keras lambda layers.

The resulting cropped and resized image can be seen in the following figure. The normalization phase is more of a numerical than a visual matter, and it's not showed below.

![Preprocessing][preprocessing]
 
####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). 
Considering that I processed every image randomly, the validation images were different from the training images, even when they came from the same original dataset
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 25).
However, I played a little bit with the number of epochs and the batch size, arriving at the conclusion that 5 epochs and a batch size of 128 was good enough.
[//]: # (that 3 epochs were enough and that a typical batch size of 128 was may be more or less the same as a batch size of 64, in terms of the loss results. That being said, I used batches of 64 images, to reduce GPU and memory usage. In addition, a low size for the batch provides better generalization of the model.)

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
[//]: # (I used a combination of center lane driving, recovering from the left and right sides of the road.)
I started out with the original data set provided by the course, analysing and preprocessing it.

For details about how I created the training data, see the next section. 

####3. Solution Design Approach

The overall strategy for deriving a model architecture was to improve a simple model progressively.

My first step was to use a convolution neural network model similar to the one found in this NVIDIA paper.
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

I thought this model might be appropriate because the objective of that paper is to set up a network that learns the entire processing
pipeline needed to steer an automobile, achieving impressive results with minimum training data and only the human steering angle as the training signal, which is the same case of this project.

First, I used the following architecture, using ReLu activations and without any dropouts. I started out with only 3 images for training: one with straight angle (zero), another with negative angle, and the other with positive angle. I increased the number of epochs until the model overfitted, that is, predicted exactly those 3 input images with extremely good precision. Then I tried other architectures like a VGG16 and a VGG19 pre-trained with 'imagenet', and others convolution neural networks comparing their resulting loss. I concluded that the NVIDIA-like model was a good choice.

Then, I added one convolutional layer at the beginning of the model with 1x1 filter and a depth of 3, applying what I learned in the previous project (Traffic Signs Classifier). The idea behind this layer is to let the network decide which color subspace fits better for this application.

After that, I duplicate the convolutional operations within each layer and added some dropout. I found that the best places to insert dropout was at the beginning and ending of the fully connected layers by trial and error with different models.
Furthermore, I added an additional layer of convolution with 128 3x3 filter.


####6. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

![Final model][model]

All max pooling layers are 2x2, and every single layer in the architecture is followed by an ELU activation (except for the preprocessing blocks, clearly).


### Analysis and augmentation of the training data

At first, the model was trained with only the original data without data augmentation. When I ran the simulator to see how well the car was driving around track one, it drove well up to the second curve (the one after the bridge) where it crashed. I thought that this could happened because most time of the training was driving straight and only a few seconds of left and right turning.

To improve the driving behavior in these cases, I decided to analyse the training data at the input in order to find out the exact amount of examples for each different steering angle. For this, I made an histogram, as we can see in the following figure:

![Original data][original_data]

We can notice that around 4500 images of the dataset (out of a total of 8000) are from the car driving straight, that is steering angle zero. This is not a good training for the model, because it will learn mostly to drive straight, but we want it to take turns as well.

I tried using left and right cameras for training, adjusting the steering angle according to NVIDIA paper, in order to obtain more turns in the training process. To achieve this, every time I read a line from the csv containing all the images paths during training, I decided randomly which camera to take with 34% chance of choosing the central camera, and 33% for each side camera. When choosing a side camera, the angle shift (let's call it LR_shift_angle) to adjust the original steering angle was a parameter to tune, but I will talk about that later. The resulting histogram can be seen in the next figure:

![Side cameras][side_cameras]

We can think of the original data histogram as a delta in the origin, because most of the images are from the car driving straight (steering angle = 0).
When we take into account left and right cameras and assume a fixed steering angle for each one (let's say +-LR_shift_angle), we can think of the result as an addition of two deltas, one in +LR_shift_angle and the other in -LR_shift_angle. The high of this deltas are proportional to the probability we are considering for taking side cameras divided by two.
So, if the straight angle delta has a height of center_camera_prob, then left and right cameras will have a height of (1-center_camera_prob)/2 each one.

As we can see, this is not a good training set, because the car will tend to drive straight or a fixed right or a fixed left steering angle. In reality, a human normally make continuous turns, rising up the steering angle smoothly. To achieve this, we need more examples of turning angles.
A good way to accomplish this is easily found on the internet, and consists on applying a random translation to every image in the horizontal axis and adjust the steering angle accordingly. However, I performed a tweak to this method, boosting up the results. The internet method (described for example in the article of Vivek Yadav)
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.afvzh0wrx
apply a uniformly distributed random translation, with a range of 0.4, so the maximum shift in the steering angle was of 0.2. This is another parameter to tune, let's call it translation_range_angle. The result of using this to augment the training data is showed in the next histogram:

![Uniform random translation][translation]

We can think of this as making the convolution between a uniform distribution with a range of "translation_range_angle", and the original data histogram.

If we combine the side cameras with the translation shifts the result we obtain can be viewed as the convolution between:
- the original data histogram plus both left and right deltas, 
and 
- a uniform distribution with a range of "translation_range_angle"

This is an addition of these 3 images (each one is the convolution of the uniform distribution with a delta):

![Side cameras and random shift][addition]

This way we see that for the resulting distribution to be uniform, we need that half of the width between deltas (that is equal to LR_shift_angle/2) match with half of the uniform range (translation_range_angle/2 = max translation_shift_angle), so that none of the uniform distributions to be added overlap and the result is also uniform.

For example, setting LR_shift_angle = 0.3 and max translation_shift_angle = 0.15 we achieve a uniform distribution. However, in this case we are making turning images from left and right cameras into really aggressive steering angles, so the driving results is not smooth.

![LR_shift_angle = 0.3, max translation_shift_angle = 0.15][LR0315]

Reducing both angles to LR_shift_angle = 0.25 and translation_shift_angle = 0.125, the driving result is smoother without loosing control.

![LR_shift_angle = 0.25, max translation_shift_angle = 0.125][LR0250125]

The drawback of this configuration is that the car is practically not able to make turns with steering angle larger than 0.25 + 0.125 = 0.375, because of the nature of the uniform distribution.
In order to overcome this problem, I came up with the idea of generating these shifts with a normal gaussian distribution instead of a uniform, allowing a continuous distribution among large steering angle. I set the mean to 0 and the standard deviation = max translation_shift_angle desired. I decided to increase a little bit this last parameter to allow larger angles even more. I end up with LR_shift_angle = 0.25 and translation_shift_angle = 0.15. The resulting histogram is showed below:

![Gaussian translation][gauss025015]

Another improvement for data augmentation was to randomly flip half of the training images and invert the corresponding steering angle, equalizing the amount of left and right turns.

To make the model more robust, I also augmented/reduced brightness randomly to simulate day and night conditions. This was fundamental in order to autonomously drive in the other track provided in the simulator. The original code for this tweak was taken from internet, but it had problems. When the brightness was reduced it worked well. However, when the brightness was augmented for pixel beyond 255, the result was not certainly what I expected. For example, when the result was 256, it was turned into 256 - 255 = 1. So I added a saturation block in the HSV space (utils.py, line 101), consisting of:

    image1[:,:,2] = np.minimum(image1[:,:,2]*random_bright, 255)

This way we make sure that if the HSV "Value" level is at maximum, then it doesn't go beyond that.

Furthermore, I also translated randomly the image in the vertical direction to simulate up and down slope of the road (this time with the uniform distribution to bound clearly this shift). For the record, I set the horizontal translation shift to 
WIDTH/20 (8 pixels) and the vertical one to HEIGHT/50 (6.4 pixels). 
[//]: # (WIDTH/3.2 (100 pixels) and the vertical one to HEIGHT/4 (40 pixels). )
This numbers are independent of the numerical value of the steering angle shift discussed above. However, it modifies the behavior the car will adopt when it sees a curve ahead. This particular values are really low, so the translation is very small for every image. The consequence of this, is that it makes the model learn that it is possible to take several steering angle values in very similar images, achieving more flexibility.

For all this tweaks to take effect, I had to iterate the original data set several times to augment significantly the training data. The original data contained around 8000 lines in the CSV log file, i.e. 8000 frames. That is 24000 images if you consider side cameras. I decided to run over the log file 8 times, taking only one image per line and applying these random tweaks, thus generating a total of 8*8000=64000 different images.

In the next figures, we can see some examples of this artificially generated images for the center camera first:

![Artificially generated images][generated_img_C]

and for the right camera next:

![Artificially generated images][generated_img_R]



### Creation of the Training Set & Training Process
### Training Process


[//]: # (The final step was to run the simulator to see how well the car was driving around track one. )
[//]: # (The car drove well up to the second curve (the one after the bridge) where it crashed. I thought that this could happened because most time of the training was driving straight and only a few seconds of left and right turning.)

[//]: # (At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.)

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
