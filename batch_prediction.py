# -*- coding: utf-8 -*-
"""
    FILE NAME: batch_prediction.py
    AUTHOR: Michalis Meyer
    DATE CREATED: 04.07.2019
    DATE LAST MODIFIED: 06.07.2019
    PYTHON VERSION: 3.6.3
    SCRIPT PURPOSE: Iterate through all images in given folder and predict class.
"""

# Imports python modules
import predict_rev3 as predict
import glob
from pathlib import Path

# Iterate through all images in folder and predict class.
for image_path in glob.glob("C:/Users/meyer-4/001_UdacityAIProgrammingwPythonFinalProject/flowers/test/99/*.jpg"):
    # Use pathlib.Path() to be able to run code on Windows.
    image_path = Path(image_path)
    # Predict class for given image.
    classes_ps, class_names = predict.predict_class(image_path, 5)

print("End of Script.")