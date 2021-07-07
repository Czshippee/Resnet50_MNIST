from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
import shutil
import gzip
import pdb
import time
from mnist_reader import *

class MNIST(VisionDataset):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(
            self,
            path: str,
            train: bool = True,
            transform: Optional[Callable] = None,
    ) -> None:
        super(MNIST, self).__init__(path, transform=transform)
        self.train = train  # training set or test set
        self.data, self.targets = self._load_data(path)

    def _load_data(self, path):
        train_data,train_label,test_data,test_label = load_mnist(path)
        if self.train:
            return train_data,train_label
        else:
            return test_data,test_label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index].reshape((28,28))
        target = int(self.targets[index])

        img = np.asarray(img, np.uint8)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('trans_img',img)
            
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

if __name__=='__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    mnist_train = MNIST(path = '/data1/scz/data/MNIST/train.csv', train = True, transform = train_transforms)
    mnist_test = MNIST(path = '/data1/scz/data/MNIST/train.csv', train = False, transform = train_transforms)
    
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=1, shuffle=True, num_workers = 8)

    for step, (x, y) in enumerate(mnist_train_loader):
        pdb.set_trace()
