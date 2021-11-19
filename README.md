# Deep-Learning-Image-Classfication-CIFAR10

## Project Aim

This is a project looking at a small image dataset. The objective is to build a convolutional neural network that performs multi-class classfication. Data is available via https://www.cs.toronto.edu/~kriz/cifar.html

I first tried to build a simple artificial neural network (ANN) to see how it performs in classifying the 10 classes of images. The initial model is built using packages TensorFlow, Keras and SciPy. I then tested a convolutional neural network (CNN) approach and evaluated the performance of each. More details of the model construction process and reasoning could be found in the Jupyter Notebook document.

### Technologies and Algorithms
* Python 3.8
* TensorFlow
* Keras
* SciPy
* Scikit-Learn
* Matplotlib
* Artificial Neural Network
* Convolutional Neural Network

## Overview
There is a total of 10 classes that we are trying to classify. Data comes in a training and testing split of 50000 and 10000 images. Each image has a pixel resolution of 32 by 32 and has three-dimension RGB intensity values. Below shows some loaded sample images and their corresponding labels.

![CIFAR10-Labels](https://user-images.githubusercontent.com/91271318/142672016-25ba4760-253c-492b-b611-e51fd563ccac.png)

![example_input](https://user-images.githubusercontent.com/91271318/142671431-30f8f2b2-6f73-4b95-8d2e-3d6c24f34c16.png)

### ANN Architecture
This shows a summary of the ANN model. Two hidden layers making up a total of 4000 neurons are used.

![ann_model](https://user-images.githubusercontent.com/91271318/142672384-7959aad7-5a24-47d5-b5a6-6b409b9caad3.png)

### ANN Results
With 10 epochs, the ANN model has relatively poor performance given low evelaution metrics for accuracy, precision, recall and f1-score. The histogram shows high variation in number of instances of each predicted class as opposed to the actual value of 1000.

![ann_classfication_report](https://user-images.githubusercontent.com/91271318/142672411-2831598f-ad24-479e-920b-2fb4eba27ae5.png)

![ann_histogram](https://user-images.githubusercontent.com/91271318/142672417-4bf7c4c4-a4ac-4fd5-be51-eda141c08513.png)


### CNN Architecture
This shows a summary of the CNN model. Two convolution+max pooling layers are used for feature extraction before the dense layers in the latter seciton.

![cnn_model](https://user-images.githubusercontent.com/91271318/142672427-a4428181-f38a-4f91-8054-a6e9c77838b7.png)

### CNN Results
With the same number of epochs, the CNN model has performed much better than the ANN model with significant improvments across all evaluation metrics.

![cnn_classfication_report](https://user-images.githubusercontent.com/91271318/142672441-79d7f551-8a96-42c4-8ea7-815d3d5749bf.png)

![cnn_histogtam](https://user-images.githubusercontent.com/91271318/142672443-921f584b-338f-406a-8b44-34d8f961b5f0.png)

## Future work options
I will be using the CIFAR-100 dataset (available via the same link) which looks at multi-label multi-class classfication. This will be more challenging than the current one but I look forward to it nonetheless.
