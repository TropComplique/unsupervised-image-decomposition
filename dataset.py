import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Images(Dataset):

    def __init__(self, folder, size):
        """
        Arguments:
            folder: a string, the path to a folder with images.
            size: an integer.
        """

        self.names = os.listdir(folder)
        self.folder = folder

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        """
        The output represents a RGB image
        with pixel values in the [0, 255] range.

        Returns:
            a byte tensor with shape [3, size, size].
        """

        name = self.names[i]
        path = os.path.join(self.folder, name)

        image = Image.open(path).convert('RGB')
        image = np.array(self.transform(image))
        return torch.from_numpy(image).permute(2, 0, 1)
