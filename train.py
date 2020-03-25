#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    FILE NAME: train.py
    AUTHOR: Michalis Meyer
    DATE CREATED: 24.05.2019
    DATE LAST MODIFIED: 07.07.2019
    PYTHON VERSION: 3.6.3
    SAMPLE COMMAND LINE: python3 train_rev1.py --file_path "C:/Users/username/path_to_project/flowers" --arch "vgg16" --epochs 5 --batch_size 64 --gpu "gpu" --running_loss True --valid_loss True --valid_accuracy True --test True
    SCRIPT PURPOSE: Train neural network.
"""

# Imports python modules
from argparse import ArgumentParser
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict

def get_input_args():
    """
    Retrieve and parse command line arguments.
    Command Line Arguments:
       - Image file path as --file_path.
       - CNN Model Architecture as --arch with default value "vgg16"
       - Number of epochs as --epochs with default value 20.
       - GPU training as --gpu.
       - Test mode to exit model training prematurely as --test.
    This function returns these arguments as an ArgumentParser object.
    Parameters:
       None - simply using argparse module to create & store command line arguments
    Returns:
       parse_args() - data structure that stores the command line arguments object.
    """
    # Create Parse using ArgumentParser
    parser = ArgumentParser()
    
    # Image file path.
    parser.add_argument("--file_path", type = str, default = "C:/Users/meyer-4/001_UdacityAIProgrammingwPythonFinalProject/flowers", help = "Image file path.")
    
    # CNN model architecture: resnet or vgg.
    parser.add_argument("--arch", type = str, default = "vgg16", help = "CNN model architecture: resnet or vgg16")
    
    # Number of epochs as --epochs with default value 5.
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of epochs. Default = 5")
    
    # Batch size as --batch_size with default value 64.
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size. Default = 64")
    
    # GPU training.
    parser.add_argument("--gpu", type = str, default = "cpu", help = "GPU training")
    
    # Model performance printing options during training.
    parser.add_argument("--running_loss", type = bool, default = True, help = "Model performance printing options during training: True / False to print running_loss.")
    
    # Model performance printing options during training.
    parser.add_argument("--valid_loss", type = bool, default = True, help = "Model performance printing options during training: True / False to print valid_loss.")
    
    # Model performance printing options during training.
    parser.add_argument("--valid_accuracy", type = bool, default = True, help = "Model performance printing options during training: True / False to print valid_accuracy.")
    
    # Testing mode.
    parser.add_argument("--test", type = bool, default = False, help = "True if in test mode to exit model training prematurely.")
    
    print("Done parsing input arguments...")
    
    return parser.parse_args()

def load_data(in_args):
    """
    Function to:
        - Specify diretories for training, validation and test set.
        - Define your transforms for the training, validation and testing sets.
        - Load the datasets with ImageFolder.
        - Using the image datasets and the trainforms, define the dataloaders.
        - Label mapping.
        
        Set size:
            Class      Total    test    train   valid
            102        8189     819     6552    818
    """
    # Specify diretories for training, validation and test set.
    data_dir = in_args.file_path
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"
    
    # Define your transforms for the training, validation, and testing sets
    # Image color channel mean:[0.485, 0.456, 0.406]. Image color channel std dev:[0.229, 0.224, 0.225]. Calculated with calculate_trainset_mean_std.py
    # Transformation on training set: random rotation, random resized crop to 224 x 224 pixels, random horizontal and vertical flip, tranform to a tensor and normalize data.
    train_transforms = transforms.Compose([transforms.RandomRotation(23),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    # Transformation on validation set: resize and center crop to 224 x 224 pixels, tranform to a tensor and normalize data.
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    # Transformation on test set: resize and center crop to 224 x 224 pixels, tranform to a tensor and normalize data.
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    global train_dataset
    global valid_dataset
    global test_dataset
    train_dataset = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
    valid_dataset = datasets.ImageFolder(data_dir + "/valid", transform=valid_transforms)
    test_dataset = datasets.ImageFolder(data_dir + "/test", transform=test_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders, as global variables.
    global batch_size
    global trainloader
    global validloader
    global testloader
    batch_size = in_args.batch_size
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # Label mapping.
    global cat_to_name
    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)
    
    print("Done loading data...")
    
    return

def build_model(in_args):
    """
    Function to:
        - Use GPU if it's available.
        - Define model.
        - Specify own classifier.
        - Specify loss function / criterion and optimizer.
    """
    # Use GPU if it's available.
    global device
    device = torch.device("cuda" if (torch.cuda.is_available() and in_args.gpu == "gpu") else "cpu")
    
    # Define model. The feature detector of the model is used. The classifier of the model is not used.
    # Available models see: https://pytorch.org/docs/master/torchvision/models.html
    global model
    model = models.vgg16(pretrained=True) if in_args.arch == "vgg16" else print("Error: No model architecture defined!")
    
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
    
    # Move your model parameters and other tensors to the device (GPU) memory when you are in PyTorch.
    model.to(device)
    
    print("Done building model...")

    return

def train_model(in_args):
    """
    Function to build and train model.
    """
    # Number of epochs.
    global epochs
    epochs = in_args.epochs
    # Set running_loss to 0
    running_loss = 0
    
    # Prepare lists to print losses and accuracies.
    global list_running_loss
    global list_valid_loss
    global list_valid_accuracy
    list_running_loss, list_valid_loss, list_valid_accuracy = [], [], []
    
    # If in testing mode, set loop counter to prematurly return to the main().
    if in_args.test == True:
        loop_counter = 0
    
    # for loop to train model.
    for epoch in range(epochs):
        # for loop to iterate through training dataloader.
        for inputs, labels in trainloader:
            # If in testing mode, increase loop counter to prematurly return to the main() after 5 loops.
            if in_args.test == True:
                loop_counter +=1
                if loop_counter == 5:
                    return
            
            # Move input and label tensors to the default device.
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Set gradients to 0 to avoid accumulation
            optimizer.zero_grad()
            
            # Forward pass, back propagation, gradient descent and updating weights and bias.
            # Forward pass through model to get log of probabilities.
            log_ps = model.forward(inputs)
            # Calculate loss of model output based on model prediction and labels.
            loss = criterion(log_ps, labels)
            # Back propagation of loss through model / gradient descent.
            loss.backward()
            # Update weights / gradient descent.
            optimizer.step()
            
            # Accumulate loss for training image set for print out in terminal
            running_loss += loss.item()
            
            # Calculate loss for verification image set and accuracy for print out in terminal.
            # Validation pass and print out the validation accuracy.
            # Set loss of validation set and accuracy to 0.
            valid_loss = 0
            valid_accuracy = 0
            
            # Set model to evaluation mode to turn off dropout so all images in the validation & test set are passed through the model.
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations.
            with torch.no_grad():
                # for loop to evaluate loss of validation image set and its accuracy.
                for valid_inputs, valid_labels in validloader:
                    # Move input and label tensors to the default device.
                    valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                    
                    # Run validation image set through model.
                    valid_log_ps = model.forward(valid_inputs)
                    
                    # Calculate loss for validation image set.
                    valid_batch_loss = criterion(valid_log_ps, valid_labels)
                    
                    # Accumulate loss for validation image set.
                    valid_loss += valid_batch_loss.item()
                    
                    # Calculate probabilities
                    valid_ps = torch.exp(valid_log_ps)
                    
                    # Get the most likely class using the ps.topk method.
                    valid_top_k, valid_top_class = valid_ps.topk(1, dim=1)
                    
                    # Check if the predicted classes match the labels.
                    valid_equals = valid_top_class == valid_labels.view(*valid_top_class.shape)
                    
                    # Calculate the percentage of correct predictions.
                    valid_accuracy += torch.mean(valid_equals.type(torch.FloatTensor)).item()
            
            # Print out losses and accuracies
            # Create string for running_loss.
            str1 = ["Train loss: {:.3f} ".format(running_loss) if in_args.running_loss == True else ""]
            str1 = "".join(str1)
            # Create string for valid_loss.
            str2 = ["Valid loss: {:.3f} ".format(valid_loss/len(validloader)) if in_args.valid_loss == True else ""]
            str2 = "".join(str2)
            # Create string for valid_accuracy.
            str3 = ["Valid accuracy: {:.3f} ".format(valid_accuracy/len(validloader)) if in_args.valid_accuracy == True else ""]
            str3 = "".join(str3)
            # Print strings
            print(f"{epoch+1}/{epochs} " + str1 + str2 + str3)
            
            # Append current losses and accuracy to lists to print losses and accuracies.
            list_running_loss.append(running_loss)
            list_valid_loss.append(valid_loss/len(validloader))
            list_valid_accuracy.append(valid_accuracy/len(validloader))
            
            # Set running_loss to 0.
            running_loss = 0
            
            # Set model back to train mode.
            model.train()
    
    print("Done training model...")
    
    return

def test_model():
    """
    Function to test model.
    """
    # Do validation on the test set
    # Prepare lists to print loss and accuracy.
    list_test_loss, list_test_accuracy = [], []
    
    # Calculate loss for test image set as well as accuracy for print out in terminal.
    # Validation pass and print out the validation accuracy.
    # Set loss of test set and accuracy to 0.
    test_loss = 0
    test_accuracy = 0
    
    # Set model to evaluation mode to turn off dropout so all images in the validation & test set are passed through the model.
    model.eval()
    
    # Turn off gradients for validation, saves memory and computations.
    with torch.no_grad():
        # for loop to evaluate loss of validation image set and its accuracy.
        for test_inputs, test_labels in testloader:
            # Move input and label tensors to the default device.
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            
            # Run test image set through model.
            test_log_ps = model.forward(test_inputs)
            
            # Calculate loss for test image set.
            test_batch_loss = criterion(test_log_ps, test_labels)
            # Accumulate loss for test image set.
            test_loss += test_batch_loss.item()
            
            # Calculate probabilities
            test_ps = torch.exp(test_log_ps)
            
            # Get the most likely class using the ps.topk method.
            test_top_k, test_top_class = test_ps.topk(1, dim=1)
            
            # Check if the predicted classes match the labels.
            test_equals = test_top_class == test_labels.view(*test_top_class.shape)
            
            # Calculate the percentage of correct predictions.
            test_accuracy += torch.mean(test_equals.type(torch.FloatTensor)).item()
            
            # Print out losses and accuracies
            print(f"Test loss: {test_loss/len(testloader):.3f} "
                  f"Test accuracy: {test_accuracy/len(testloader):.3f} ")
            
            # Append current losses and accuracy to lists to print losses and accuracies.
            # list_test_loss.append(test_loss/len(testloader))
            # list_test_accuracy.append(test_accuracy/len(validloader))
            list_test_loss.append(test_loss/len(testloader))
            list_test_accuracy.append(test_accuracy/len(validloader))
            
            # Set model back to train mode.
            model.train()
            
            print("Done testing model...")
            
            return

def save_checkpoint():
    """
    Function to save checkpoint.
    """
    # Save the checkpoint.
    
    # Build the checkpoint dictionary with additional details in checkpoint dictionary "checkpoint.pth".
    # Reference: https://medium.com/@tsakunelsonz/loading-and-training-a-neural-network-with-custom-dataset-via-transfer-learning-in-pytorch-8e672933469
    # Reference: https://towardsdatascience.com/load-that-checkpoint-51142d44fb5d
    checkpoint = {"model": models.vgg16(pretrained=True),
                  "input_size": 2208,
                  "output_size": 102,
                  "epochs": epochs,
                  "batch_size": batch_size,
                  "state_dict": model.state_dict(),
                  "state_features_dict": model.features.state_dict(),
                  "state_classifier_dict": model.classifier.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "criterion_state_dict": criterion.state_dict(),
                  "class_to_idx": train_dataset.class_to_idx,
                 }
    
    # Save the checkpoint dictionary.
    torch.save(checkpoint, 'checkpoint.pth')
    
    # Load parameters.
    ckpt = torch.load('checkpoint.pth')
    ckpt.keys()

    print("Done saving checkpoint...")
    
    return

def main():
    """
    Main Function
    """
    
    # Assigns variable in_args to parse_args()
    in_args = get_input_args()
    # Load data.
    load_data(in_args)
    # Build model.
    build_model(in_args)
    # Train model
    # I am using so many global variables because otherwise following command becomes to long and gives me a hard time to get to run.
    train_model(in_args)
    # Test model.
    test_model()
    # Save checkpoint.
    save_checkpoint()
    # Script end.
    print("End of Script.")

# Run main function.
if __name__ == '__main__':
    main()
