# Introduction 

In deepchecks there are multiple test suites already available which can check the train vs test model metrics , datasets characteristics and  much more. One example of such suite is DatasetsSizeComparison.

DatasetsSizeComparison
Conditions:
0: Test-Train size ratio is greater than 0.01

These suites contains conditions which can also be edited , changed or new added. But this preavialable suite contains codition which checks that test data and train data ratio is greater than 0.01.

## Installation
Deepchecks package can be installed on your local machine via pip or in your Jupyter notebook section using the following command:

```Python
import sys
!{sys.executable} -m pip install "deepchecks[vision]" --quiet --upgrade # --user
```

## Types
So there are primarily three types of checks taking place in Deepcheck in the process. They are as follows :
1) Data Integrity Check 
2) Check for distribution of data for train and test
3) Model Performance Evaluation for unseen data or close to real-world data

Dataset_integrity checks the integrity present in the dataset such as missing fields and etc.
Train Test validation set checks the correctness of the split of the data for training and testing phase.
Lastly model evaluation cross-checks the model performance and genericness and also the signs of overfitting if present.

## Object Detection
In computer vision, object detection is one the most fundamental applications and arguably the most important one. Hence a good testing suite should be able to quantify the results of object detection.

In this example, we will try to detect tomatoes in images using a pre-trained model and then test its accuracy using deepchecks.

First, lets import the desired libraries:

```Python
import os
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from functools import partial

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

from deepchecks.vision.detection_data import DetectionData
```