import os
import numpy as np
from PIL import Image


class DataProcess(object):
    def __init__(self, path, **kwargs):
        self.path = path
        self.img_ids = [i for i in os.listdir(self.path) if i.endswith('.jpg')]
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.img_list = [os.path.join(self.path, i) for i in self.img_ids]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img = (img / 255. - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        data = {
            'img': img,
            'img_path': img_path
        }
        return data
