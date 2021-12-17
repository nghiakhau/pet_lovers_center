import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
import os
from torchvision import transforms
import matplotlib.pyplot as plt


def show_image(filepath, img_size=224):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(
        src=img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    return plt.imshow(img)


def show_transformed_image(filepath, img_size=224):
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=45),
                    transforms.PILToTensor(),
                    ])
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(
        src=img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    img = transform(img)
    img = np.moveaxis(img.numpy(), 0, -1)
    return plt.imshow(img)


# img = np.random.choice(glob.glob('data/petfinder-pawpularity-score/train/*'))
# show_image(img)
# show_transformed_image(img)


# Pretrained Data Transforms
TRANSFORM_PRE = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])

# Non-pretrained Data Transforms
TRANSFORM_NOPRE = transforms.Compose([transforms.ToTensor()])

train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.PILToTensor()])

test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.PILToTensor()])


def get_image(
        path: str, size: int, transform: transforms.Compose) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    img = cv2.resize(
        src=img, dsize=(size, size), interpolation=cv2.INTER_AREA)
    if transform:
        img = transform(img)
    else:
        # (H, W, C) => (C, H, W)
        img = np.moveaxis(img, -1, 0)
    return img


def transform_binary_features(features):
    for i, x in enumerate(features):
        features[i] = (i * 2) + x
    return features


class PetDataset(Dataset):
    def __init__(
            self, data_dir, img_dir, img_size, test=False, transform=None):
        # "data/petfinder-pawpularity-score/test.csv"

        data = pd.read_csv(data_dir)
        self.img_names = data['Id'].copy().values
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        if test:
            self.xs = data.iloc[:, 1:].copy().values
        else:
            self.xs = data.iloc[:, 1:-1].copy().values
        self.xs = np.apply_along_axis(transform_binary_features, 1, self.xs)
        self.ys = data.iloc[:, -1].copy().values/100

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx] + ".jpg")
        img = get_image(img_path, self.img_size, self.transform)

        return {
            'img': img,
            'x': self.xs[idx],
            'y': self.ys[idx]}
