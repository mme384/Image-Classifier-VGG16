#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    FILE NAME: predict.py
    AUTHOR: Michalis Meyer
    DATE CREATED: 25.05.2019
    DATE LAST MODIFIED: 06.07.2019
    PYTHON VERSION: 3.6.3
    SCRIPT PURPOSE: Load image to predict class and probability.
"""

# Imports python modules
import torch
from torch import nn
from torch import optim
from torchvision import models
import json
from collections import OrderedDict
from PIL import Image
from math import floor, ceil
import numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rebuild_model(filepath):
    """
    Load checkpoint and rebuilding model.
    """
    
    # Label mapping.
    global cat_to_name
    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)
    
    # Use GPU if it's available.
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define model. The feature detector of the model is used. The classifier of the model is not used.
    # See models: https://pytorch.org/docs/master/torchvision/models.html
    global model
    model = models.vgg16(pretrained=True)
    
    # Freeze model parameter so backpropagate through not necessary here.
    for param in model.parameters():
        param.requires_grad = False
    
    # Specify own classifier.
    model.classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(25088, 4096)),
        ("relu1", nn.ReLU()),
        ("dropout1", nn.Dropout(0.2)),
        ("fc2", nn.Linear(4096, 512)),
        ("relu2", nn.ReLU()),
        ("dropout2", nn.Dropout(0.2)),
        ("fc3", nn.Linear(512, 102)),
        ("output", nn.LogSoftmax(dim=1)),
    ]))
    
    # Specify the loss function.
    global criterion
    criterion = nn.NLLLoss()
    
    # Specify the optimizer to optimize the weights and biases.
    # Only train the classifier parameters. The feature detector parameter are frozen.
    global optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    
    # Load checkpoints.
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # print("Done rebuilding model...")
    
    return

def process_image(image_path):
    """
    Process image.
    """
    
    # Load image
    image = Image.open(image_path)
    
    # Resize image keeping aspect ratio.
    # Reference: https://stackoverflow.com/questions/4321290/how-do-i-make-pil-take-into-account-the-shortest-side-when-creating-a-thumbnail
    # Define new size
    size = 255
    # Get size and calculate new height keeping the aspect ratio.
    width, heigth = image.size
    ratio = float(heigth) / float(width)
    newheigth = int(floor(size*ratio))
    # Resize.
    image = image.resize((size, newheigth), Image.NEAREST)
    
    # Center crop image to 224 x 224 pixels.
    # Define new size.
    size = 224
    # Get size and calculate image coordinates of the box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
    width, heigth = image.size
    coordinate1 = int(ceil((width - size)/2))
    coordinate2 = int(floor((heigth - size)/2))
    coordinate3 = width - int(floor((width - size)/2))
    coordinate4 = heigth - int(floor((heigth - size)/2))
    # Crop.
    image = image.crop((coordinate1, coordinate2, coordinate3, coordinate4))
     
    # Convert image color channels from 0-255 to 0-1.
    # Convert PIL image to Numpy array.
    image = np.array(image)
    image = image/255
    
    # Normalize image.
    # Specify mean & std. dev. and normalize image.
    mean = [0.485, 0.456, 0.406]
    stddev = [0.229, 0.224, 0.225]
    image = (image - mean) / stddev
    
    # Reorder dimensions so the order of color channels is as expected.
    image = image.transpose((2, 0, 1))
    
    # print("Done processing image...")
    
    return image

def predict(image_path, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    Resources: https://github.com/WenjinTao/aipnd-project/blob/master/predict.py
    """
    
    # Load and process image.
    image_processed = process_image(image_path)
    
    # Rearrange image to 1D row vector.
    image_processed = np.expand_dims(image_processed, 0)
    
    # Convert image from np.array to tensor.
    image_processed = torch.from_numpy(image_processed)
    
    # Set model to evaluation mode to turn off dropout so all images in the validation & test set are passed through the model.
    model.eval()
    
    # Turn off gradients for validation, saves memory and computations.
    with torch.no_grad():
        
        # Move model to cuda.
        model.to(device)
        
        # Move input tensors to the default device.
        input = image_processed.to(device, dtype=torch.float)
        
        # Run image through model.
        log_ps = model.forward(input)
        
        # Calculate probabilities
        ps = torch.exp(log_ps)
        
        # Get the most likely class using the ps.topk method.
        classes = ps.topk(topk, dim=1)
    
    # Set model back to train mode.
    model.train()
    
    # Extract predicted class probabilities, copy tensor to CPU, convert to list and flatten list".
    classes_ps = classes[0]
    classes_ps = classes_ps.cpu().tolist()
    classes_ps = [item for sublist in classes_ps for item in sublist]
    
    # Extract predicted class index, copy tensor to CPU, convert to list.
    classes_idx = classes[1]
    classes_idx = classes_idx.cpu().tolist()
    
    # Get predicted flower names from cat_to_name
    class_names = [cat_to_name.get(str(idx)) for idx in np.nditer(classes_idx)]
    
    print("Class Index: ", classes_idx)
    print("Class Names: ", class_names)
    print("Class Probabilities: ", classes_ps)
    
    # print("Done predicting...")
    
    return classes_ps, class_names, ps, classes


def imshow(image, ax=None, title=None):
    """
    Display image.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        # image = image.numpy().transpose((1, 2, 0))
        image = image.transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.603, 0.603, 0.603])
        std = np.array([0.227, 0.227, 0.227])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        # print("Done showing image...")
        
        return ax


def display_results(image_path, classes_ps, class_names):
    """
    Display image and prediction.
    """
    
    # Convert pathlib.Path to string to be able to run code on Windows.
    image_path = str(image_path)
    # Read image
    image = mpimg.imread(image_path)    
    
    # Define subplot 1.
    plt.subplot(2, 1, 1)
    
    # Plot image.
    plt.imshow(image)
    plt.axis('off')
    
    # Define subplot 2.
    plt.subplot(2, 1, 2)
    
    # Plot horizontal bar chart with probabilities and flower names.
    plt.barh(class_names, classes_ps)
    
    # Show plot.
    plt.show()
    
    # print("Done displaying results...")
    
    return


def predict_class(image_path, topk=5):
    """
    Main function to predict class.
    """
    
    # Rebuild model.
    rebuild_model("checkpoint.pth")
    # Process image.
    image_processed = process_image(image_path)
    # Predict class and probaility for image.
    classes_ps, class_names, ps, classes = predict(image_path, topk)
    # Display result
    # display_results(image_path, classes_ps, class_names)
    # Separator
    print("********************************************")
    
    return classes_ps, class_names
