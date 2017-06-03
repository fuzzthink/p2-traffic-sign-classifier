+++
showonlyimage = false
draft = false
image = "img/posts/trafficsigns-thumb.png"
date = "2017-05-31T18:25:22+05:30"
title = "Traffic Sign Classifier"
weight = 2
+++

In this [project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project), we will design and implement a Deep Convolution Neural Networks that learns to recognize and classify traffic signs. Here is an example of a [published baseline model](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) on this problem by Pierre Sermanet and Yann LeCun. The goals and various aspects to consider in this project are:

* Design of the Neural Network architecture
* Explore and visualize the data set
* Train, optimize, and test a model architecture
* Try various preprocessing techniques
* Data Augmentation/Generation
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results


[//]: # (Image References)

[img1]: pngs/trainimages-hist.png
[img2]: pngs/trainimages-samples.png
[img3]: pngs/testimages.png
[img4]: pngs/testresult-vis.png


---
### Environment Setup

The environment can be created with [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

---
### Dataset

The 118MB zipped dataset used for this project is downloaded from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)

It is a 32x32 resized pickled version of the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Place the unzipped dataset into `../data/` as the project will expect it from there.

The pickled data is a dictionary with 4 key/value pairs:

- 'features': 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- 'labels': 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
- 'sizes': list containing tuples, (width, height) representing the the original width and height the image.
- 'coords': list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. __NOTE:__ These coordinates assume the original image. The pickled data contains resized versions (32x32) of these images.

---
### Source Files 
- `train.py` - Train and save model
- `test.py` - Test model
- `Traffic_Sign_Classifier.ipynb` - Complete project in jupyter notebook. It includes additional visualizations. Older version of the dataset is needed. It should work with the [Nov'16 dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip) or [Dec'16 dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

To run:
```sh
python train.py
python test.py
jupyter notebook Traffic_Sign_Classifier.ipynb  # for additional visualizations
open Traffic_Sign_Classifier.html # to see static version of notebook 
```


---
### Data Exploration

The dataset is split into 75% training and 25% test.

- Number of training examples: 39209
- Number of testing examples: 12630
- Image data shape: (32, 32, 3)
- Number of classes: 43
- Distributions of classes: 210 2220 2250 1410 1980 1860  420 1440 1410 1470 2010 1320 2100 2160  780 630  420 1110 1200  210  360  330  390  510  270 1500  600  240  540  270 450  780  240  689  420 1200  390  210 2070  300  360  240  240

Distributions of classes Visualized:
![][img1]

*Fig 1. Traffic sign class distributions in dataset*


![][img2]
*Fig 2. A sample from each class of traffic signs*


---
### Preprocessing

First, the images are converted to grayscale since there isn't any two classes of traffic signs that are different based solely on color. This conversion will speedup the training and likely also lessen the chance of over-fitting. 

Next, they are normalized to values between -.5 and .5 so the weights don't blow up.

Since the data are images, it is a good idea to augment the data with flip, rotation, and shift operations to better model real world images of traffic signs. This is done via `keras.preprocessing.image.ImageDataGenerator`. Parameters passed are:

```python
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True)
```


The way this generator works is not by producing extra data set, but by randomly select parameters within the specified range and apply to all original images in the training set. After an epoch is completed, the next epoch is started and training data is once again augmented by applying specified transformations randomly to the original training data.

---
### Convolutional Neural Network Architecture

The model I chose is the Vgg model [Vgg model sample from Keras.io](https://keras.io/getting-started/sequential-model-guide/#examples). It was chosen since I wanted to use Keras and the model is one of the well known and simple to understand models.

Since traffic signs are not really different from general images and Vgg is shown to be both simple and work very well on general image sets, it should fit the problem set nicely.

The final model is consisted of the following layers:

(all strides are 1x1, except for max pooling which is 2x2)

| Layer             | Output/Description                    | 
|:-----------------:|:---------------------------------------------:| 
| Input             | 32x32 x1 Grayscale image                | 
| Convolution 3x3   | 30x30 x32, valid padding, RELU activation |
| Convolution 3x3   | 28x28 x32, valid padding, RELU activation |
| Max pooling 2x2   | 14x14 x32 |
| Dropout           | 25%       |
| Convolution 3x3   | 12x12 x64, valid padding, RELU activation |
| Convolution 3x3   | 10x10 x64, valid padding, RELU activation |
| Max pooling 2x2   | 5x5 x64 |
| Dropout           | 25%     |
| Flatten           | 1600    |
| Fully connected   | 512, RELU activation |
| Dropout           | 50%     |
| Fully connected   | 43, softmax activation |
| Accuracy          | training: 97.78%, validation: 99.76%, test: 97.11%

The different Convolution layers act as feature extractors. The lower layers for lower level features. For upper layers, since there are many high level features needed to classify the images, an increase in number of filters (that takes less number of input parameters) is needed to correctly classify the image set.

The model did very well. So well in fact that none of my endless adjustments outperforms it more than a margin of error. The final model I used only changes the Fully connected layer from 256 to 512 weights. Basically, this change did not affect the accuracies too much, slightly better on training and validation and splitting hairs worst on test. 

The parameters I tried to tuned includes adding and removing Convolution layers, numbers of filters in various Convolution layers, same/valid padding, dropout %, and final fully connected layers. 

Out of the many modifications, two other architectures are documented below, along with the unmodified Vgg model. 

Unmodified Vgg:

| Layer             | Output/Description                    | 
|:-----------------:|:---------------------------------------------:| 
| Input             | 32x32 x1 Grayscale image                | 
| Convolution 3x3   | 30x30 x32, valid padding, RELU activation |
| Convolution 3x3   | 28x28 x32, valid padding, RELU activation |
| Max pooling 2x2   | 14x14 x32 |
| Dropout           | 25%       |
| Convolution 3x3   | 12x12 x64, valid padding, RELU activation |
| Convolution 3x3   | 10x10 x64, valid padding, RELU activation |
| Max pooling 2x2   | 5x5 x64 |
| Dropout           | 25%     |
| Flatten           | 1600    |
| Fully connected   | 256, RELU activation |
| Dropout           | 50%     |
| Fully connected   | 43, softmax activation |
| Accuracy          | training: 97.24%, validation: 99.58%, test: 97.14%

2 other models I used worth documenting:

| Layer             | Output/Description                    | 
|:-----------------:|:---------------------------------------------:| 
| Input             | 32x32 x1 Grayscale image                | 
| Convolution 3x3   | 32x32 x32, same padding, RELU activation |
| Convolution 3x3   | 30x30 x32, valid padding, RELU activation |
| Max pooling 2x2   | 15x15 x32 |
| Dropout           | 20%       |
| Convolution 3x3   | 15x15 x64, same padding, RELU activation |
| Convolution 3x3   | 13x13 x64, valid padding, RELU activation |
| Max pooling 2x2   | 6x6 x64 |
| Dropout           | 20%       |
| Convolution 3x3   | 6x6 x128, same padding, RELU activation |
| Convolution 3x3   | 4x4 x128, valid padding, RELU activation |
| Max pooling 2x2   | 2x2 x128  |
| Dropout           | 20%       |
| Flatten           | 512 |
| Fully connected   | 512, RELU activation |
| Dropout           | 50%     |
| Fully connected   | 43, softmax activation |
| Accuracy          | training: 98.73%, validation: 99.52%, test: 96.25%

| Layer             | Output/Description                    | 
|:-----------------:|:---------------------------------------------:| 
| Input             | 32x32 x1 Grayscale image                | 
| Convolution 3x3   | 32x32 x32, same padding, RELU activation |
| Max pooling 2x2   | 16x16 x32 |
| Dropout           | 20%       |
| Convolution 3x3   | 16x16 x64, same padding, RELU activation |
| Max pooling 2x2   | 8x8 x64 |
| Dropout           | 20%       |
| Convolution 3x3   | 8x8 x128, same padding, RELU activation |
| Max pooling 2x2   | 4x4 x128  |
| Dropout           | 20%       |
| Flatten           | 2048 |
| Fully connected   | 512, RELU activation |
| Dropout           | 50%  |
| Fully connected   | 43, softmax activation |
| Accuracy          | training: 95.32%, validation: 99.20%, test: 94.14%


Had training not take such a long time, I would like to try other different architectures.

---
### Validation and Test

Validation:
```
...
Epoch 28/30
29406/29406 [==============================] - 4s - loss: 0.0804 - acc: 0.9751 - val_loss: 0.0102 - val_acc: 0.9979
Epoch 29/30
29406/29406 [==============================] - 4s - loss: 0.0757 - acc: 0.9757 - val_loss: 0.0129 - val_acc: 0.9965
Epoch 30/30
29406/29406 [==============================] - 4s - loss: 0.0704 - acc: 0.9778 - val_loss: 0.0099 - val_acc: 0.9976
```

Test result:

    loss: 0.09800048008278991
    acc: 0.978701504354711

---
### Further Testing and Visualization

Five additional images were used for further testing and analysis. 

![][img3]

*Fig 3. Additional images used for testing*

These images were found in Google Image search. Since the test images are already German traffic signs the training hasn't seen before, I decided to find US traffic sign images to see how the predictions will be like. The images have been chosen such that:

* 2 of the 5 (No Passing and Road Work) are in the training set's 43 classes. 
* 1 of the 5 (Yield) is in the 43 classes, but is slightly different as stated above.
* 2 of the 5 (No Pedestrian, No Exit) are not in the training set.

#### Result

    Sign          | Prediction
    No Pedestrian | 100.00%:b'Road work'            | 00.00%:b'Double curve'          | 00.00%:b'Keep left'             
    ______________| 00.00%:b'Wild animals crossing' | 00.00%:b'Right-of-way at the nex'

    Do Not Enter  | 99.89%:b'No passing'            | 00.10%:b'Yield'                 | 00.00%:b'Slippery road'         
    ______________| 00.00%:b'No entry'              | 00.00%:b'Stop'                  

    No Exit       | 41.63%:b'Speed limit (80km/h)'  | 23.88%:b'Speed limit (50km/h)'  | 12.39%:b'Speed limit (100km/h)' 
    ______________| 09.21%:b'Stop'                  | 06.21%:b'Speed limit (30km/h)'  

    Road Work     | 45.43%:b'Traffic signals'       | 24.47%:b'Speed limit (70km/h)'  | 08.30%:b'Speed limit (30km/h)'  
    ______________| 06.05%:b'Children crossing'     | 05.52%:b'General caution'       

    Yield         | 95.11%:b'Yield'                 | 03.89%:b'Wild animals crossing' | 00.67%:b'Bumpy road'            
    ______________| 00.33%:b'Bicycles crossing'     | 00.00%:b'General caution'       

![][img4]

*Fig 4. Test Prediction Result*


#### Result summary

| Image             |     Prediction                                | 
|:-------------:|:---------------------------------------------:| 
| No Pedestrain | Off, as expected as nothing close in training | 
| Do Not Enter  | 100% No Passing, correct                                          |
| No Exit       | Off, as expected as nothing close in training |
| Road Work     | 100%, correct                      |
| Yield         | 18% Yield, German's red border is much thinner, so 18% is a decent accuracy |

3 out of 5 of the predictions are off, which is expected as the signs between US and Germany are quite different.

Since only 3 out of 5 images are part of the 43 classes, we should only include only these 3 for the comparison.

The model was very confident of the No Passing and Road Work images, with 100% probability on them.
It shows it does indeed reflect the model's accuracy on the test data.

Its 82% probability of guessing Bumpy Road seems to suggest the weight of having "something" inside the white triangle is greater than the orientation of the triangle.

---

##### [Project Review](UdacityReviews.pdf)
