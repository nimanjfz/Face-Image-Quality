from torch.utils.data import Dataset, DataLoader
import glob
import random
import numpy as np
import os
import cv2
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import pandas as pd


# Define Dataset class
class CmuDataset(Dataset):
    def __init__(self, img_dir, img_labels, transform):
        self.img_dir = img_dir
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image_path = self.img_dir[idx]
        image = cv2.imread(image_path)
        image = transform(image)
        label = self.img_labels[idx]
        return image, label


# Creating Train set
data = pd.read_csv('./dataset/image_quality_scores.txt', header=None)
train_image_paths = data.loc[:,0]
train_image_labels = data.loc[:,1]


transform = transforms.Compose([
    # mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# Creating Dataset and DataLoader for training
train_dataset = CmuDataset(train_image_paths, train_image_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=2600, shuffle=True, num_workers=5)

#Creating Test set
test_data_path = './dataset/test'
test_image_paths = []
test_image_labels = []

for data_path in glob.glob(test_data_path + '/**/*.png', recursive=True):
    if (data_path.split('/')[-1].split('.png')[0].split('_')[-2] == '00'):
        test_image_paths.append(data_path)
        test_image_labels.append(np.float32(data_path.split('/')[-1].split('_')[6].split('.png')[0]))

# Creating Dataset and DataLoader for test
test_dataset = CmuDataset(test_image_paths, test_image_labels, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


