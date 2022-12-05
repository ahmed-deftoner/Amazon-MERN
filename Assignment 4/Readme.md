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

The model is trained on a dataset consisted of 895 images of tomatoes, with bounding box annotations provided in PASCAL VOC format. All annotations belong to a single class: tomato.

The dataset is available at the following link: https://www.kaggle.com/andrewmvd/tomato-detection

Now we will pre-process the images

```Python
url = 'https://figshare.com/ndownloader/files/34488599'
urllib.request.urlretrieve(url, 'tomato-detection.zip')

with zipfile.ZipFile('tomato-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

class TomatoDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.images = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, 'annotations'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        bboxes = []
        labels = []
        with open(ann_path, 'r') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    continue
                cls_id = 1
                xmlbox = obj.find('bndbox')
                b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymax').text)]
                bboxes.append(b)
                labels.append(cls_id)

        bboxes = torch.as_tensor(np.array(bboxes), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)

        if self.transforms is not None:
            res = self.transforms(image=np.array(img), bboxes=bboxes, class_labels=labels)

        target = {
            'boxes': [torch.Tensor(x) for x in res['bboxes']],
            'labels': res['class_labels']
        }

        img = res['image']

        return img, target

    def __len__(self):
        return len(self.images)

data_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

dataset = TomatoDataset(root=os.path.join(os.path.curdir, 'tomato-detection/data'),
                        transforms=data_transforms)
train_set, test_set = torch.utils.data.random_split(dataset,
                                                    [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)],
                                                    generator=torch.Generator().manual_seed(42))
test_set.transforms = A.Compose([ToTensorV2()])
train_loader = DataLoader(train_set, batch_size=64, collate_fn=(lambda batch: tuple(zip(*batch))))
test_loader = DataLoader(test_set, batch_size=64, collate_fn=(lambda batch: tuple(zip(*batch))))
```

Lets visualize a few images:

```Python
def prepare(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) * 255
    inp = inp.transpose((2,0,1))
    return torch.tensor(inp, dtype=torch.uint8)

import torchvision.transforms.functional as F


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(20,20))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

from torchvision.utils import draw_bounding_boxes

data = next(iter(train_loader))
inp, targets = data[0][:4], data[1][:4]


result = [draw_bounding_boxes(prepare(inp[i]), torch.stack(targets[i]['boxes']),
                              colors=['yellow'] * torch.stack(targets[i]['boxes']).shape[0], width=5)
          for i in range(len(targets))]
show(result)
```

Next, we will download a pre-trained SSDlite model and a MobileNetV3 Large backbone from the official PyTorch repository.

After downloading the model, we will fine-tune it for our particular classes. We will do it by replacing the pre-trained head with a new one that matches our needs.

```Python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
num_anchors = model.anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
model.to(device)
```