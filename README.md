# AI Programming with Python Project

This project contains the code created as partial fulfilment for Udacity's AI Programming with Python Nanodegree program.

First, train.py is developed to define a CNN VGG16 in PyTorch, trained and the network parameters stored in a checkpoint file. Second, predict.py rebuilds the model and predict the class of an input image

## PROJECT FILES:
- README.md: Project README
- train.py: Define and train the CNN
- cat_to_name.json: json file containing the categories (predicted classes)
- predict.py: Rebuild the model and predict the class of an input image
- batch_prediction.py: Run predict.py on a batch of images
- calculate_trainset_mean_std.py: Calculate the images mean and standard deviation for preparing the images for training

## LINK DATA SET:
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

## MODEL
The model VGG16 provided by PyTorch is used. The pretrained feature detector is used without modifications, but not the pretrained classifier. The custom classifier is used.
- Input layer has 25088 inputs
- The hidden layers have 4896 and 512 nodes respectively, use ReLU as the activation function and the dropout layers with 20% dropout rate.
- The output layer has 102 nodes, as there are 102 classes, and uses Softmax as the activation function.

## PYTHON VERSION
3.7

## PREREQUISITS ON PYTHON Modules
- torch
- torchvision
- json
- collection
- PIL
- math
- numpy
- matplotlib

## KNOWN BUGS
No known bugs
