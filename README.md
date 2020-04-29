# AI Programming with Python Project

CNN VGG16 image classifier build on PyTorch. Developed as partial fulfilment for Udacity's AI Programming with Python Nanodegree program.

1) train.py defines a CNN based on VGG16 based on PyTorch, traines the model and saves the checkpoints.
2) predict.py rebuilds the model and predict the class of an input image

## Project Files
- README.md: Project README
- train.py: Define and train the CNN
- cat_to_name.json: json file containing the categories (predicted classes)
- predict.py: Rebuild the model and predict the class of an input image
- batch_prediction.py: Run predict.py on a batch of images
- calculate_trainset_mean_std.py: Calculate the images mean and standard deviation for preparing the images for training

## Data Set
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

## Model
The model VGG16 provided by PyTorch is used. The pretrained feature detector is used without modifications.

The custom classifier is used instead of the pretrained classifier. The classifier has following structure:
- Input layer has 25088 inputs
- The hidden layers have 4896 and 512 nodes respectively, use ReLU as the activation function and the dropout layers with 20% dropout rate.
- The output layer has 102 nodes, as there are 102 classes, and uses Softmax as the activation function.

## Python Version
3.7

## Python Modules
- torch
- torchvision
- json
- collection
- PIL
- math
- numpy
- matplotlib

## Bugs
No known bugs

## MIT License

Copyright (c) 2018 Udacity

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

