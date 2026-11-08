import os
import random
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


def encode_mask(gt):
    gt = np.array(gt)
    return (gt > 127).astype(np.float32)[None, ...]


def get_preprocess_fn(encoder_name, encoder_weights='imagenet'):
    weights = encoder_weights
    if weights is None or str(weights).lower() in ('', 'none', 'null'):
        weights = 'imagenet'
    return smp.encoders.get_preprocessing_fn(encoder_name, weights)


class MyDataset(Dataset):
    def __init__(self, img_dir, gt_dir, img_size=512, augment=False, preprocessing_fn=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.augment = augment
        self.preprocessing_fn = preprocessing_fn
        self.img_list = sorted(os.listdir(img_dir))

        print(f"img_files size={len(self.img_list)}, mask_files size={len(self.img_list)}", file=sys.stderr)

    def __len__(self):
        return len(self.img_list)

    def _load_pair(self, idx):
        name = self.img_list[idx]
        image = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        gt = Image.open(os.path.join(self.gt_dir, name)).convert("L")

        image = TF.resize(image, [self.img_size, self.img_size])
        gt = TF.resize(gt, [self.img_size, self.img_size], interpolation=Image.NEAREST)
        return image, gt

    def __getitem__(self, idx):
        image, gt = self._load_pair(idx)

        if self.augment:
            if random.random() < 0.5:
                image, gt = TF.hflip(image), TF.hflip(gt)
            if random.random() < 0.5:
                image, gt = TF.vflip(image), TF.vflip(gt)
            angle = random.uniform(-20, 20)
            image, gt = TF.rotate(image, angle), TF.rotate(gt, angle)

        if self.preprocessing_fn is not None:
            image = self.preprocessing_fn(np.array(image))
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        else:
            image = transforms.ToTensor()(image)

        gt = torch.from_numpy(encode_mask(gt))

        return image, gt
