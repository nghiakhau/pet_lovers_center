import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
import os
from torchvision import transforms


# Pretrained Data Transforms
TRANSFORM_PRE = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])

# Non-pretrained Data Transforms
TRANSFORM_NOPRE = transforms.Compose([transforms.ToTensor()])

train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor()])

test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.ToTensor()])


def get_image(path: str, size: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        src=image, dsize=(size, size), interpolation=cv2.INTER_AREA)
    image_size, _, chanels = image.shape
    return image.reshape((chanels, image_size, image_size))


def transform_binary_features(features):
    for i, x in enumerate(features):
        features[i] = (i * 2) + x
    return features


class PetDataset(Dataset):
    def __init__(
            self, data_dir, img_dir, image_size, test=False, transform=None):
        # "data/petfinder-pawpularity-score/test.csv"

        data = pd.read_csv(data_dir)
        self.img_names = data['Id'].copy().values
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transform
        if test:
            self.xs = data.iloc[:, 1:].head(100).copy().values
        else:
            self.xs = data.iloc[:, 1:-1].head(100).copy().values
        self.xs = np.apply_along_axis(transform_binary_features, 1, self.xs)
        self.ys = data.iloc[:, -1].head(100).copy().values/100

    def __len__(self):
        # return len(self.img_names)
        return 100

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx] + ".jpg")
        img = get_image(img_path, self.image_size)
        if self.transform:
            img = self.transform(img)

        return {
            'img': img,
            'x': self.xs[idx],
            'y': self.ys[idx]}
