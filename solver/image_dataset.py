#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from PIL import Image
from torch.utils import data


class DataSet(data.Dataset):
    """Characterizes a dataset for PyTorch
    """
    def __init__(self, args, examples, labels, attr, transform, is_train):
        # Initialization
        self.labels = labels
        self.examples = examples
        self.attrs = attr
        self.transform = transform
        self.image_dir = args['image_dir']
        self.args = args
        # self.n_classes = self.args['n_classes']
        self.is_train = is_train

    def __len__(self):
        # Denotes the total number of samples
        return len(self.examples)

    def __getitem__(self, idx):
        # Generates one sample of data
        id = self.examples[idx].strip()
        # Convert to RGB to avoid png.
        X = Image.open(self.image_dir + id).convert('RGB')
        X = self.transform(X)
        label = self.labels[id]
        if self.attrs is None:
            return X, label
        else:
            attr = self.attrs[id]
            return X, label, attr


class Data2Set(data.Dataset):
    """Characterizes a dataset for PyTorch
    """
    def __init__(self, args, examples, labels, attr, transform, is_train):
        # Initialization
        self.labels = labels
        self.examples = examples
        self.attrs = attr
        self.transform = transform
        self.image_dir = args['image_dir']
        self.args = args
        # self.n_classes = self.args['n_classes']
        self.is_train = is_train

    def __len__(self):
        # Denotes the total number of samples
        return len(self.examples)

    def __getitem__(self, idx):
        # Generates one sample of data
        [id, dty] = self.examples[idx].strip()
        # Convert to RGB to avoid png.
        X = Image.open(self.image_dir + id).convert('RGB')
        X = self.transform(X)
        label = self.labels[id]
        if self.attrs is None:
            return X, label, dty
        else:
            attr = self.attrs[id]
            return X, label, attr, dty


class DistillDataSet(data.Dataset):
    """Characterizes a dataset for PyTorch
    """
    def __init__(self, examples, labels, att=None):
        # Initialization
        self.labels = labels
        self.examples = examples
        self.att = att

    def __len__(self):
        # Denotes the total number of samples
        return len(self.examples)

    def __getitem__(self, idx):
        # Generates one sample of data
        # Convert to RGB to avoid png.
        X = self.examples[idx]
        label = self.labels[idx]
        if self.att is not None:
            att = self.att[idx]
            return X, label, att
        else:
            return X, label


class DistillTrainDataSet(data.Dataset):
    """Characterizes a dataset for PyTorch
    """
    def __init__(self, args, examples, labels, transform):
        # Initialization
        self.labels = labels
        self.examples = examples
        self.image_dir = args['image_dir']
        self.transform = transform

    def __len__(self):
        # Denotes the total number of samples
        return len(self.examples)

    def __getitem__(self, idx):
        # Generates one sample of data
        # Convert to RGB to avoid png.
        id = self.examples[idx].strip()
        X = Image.open(self.image_dir + id).convert('RGB')
        X = self.transform(X)
        label = self.labels[idx]
        return X, label


class DistillTestDataSet(data.Dataset):
    """Characterizes a dataset for PyTorch
    """
    def __init__(self, args, examples, labels, transform):
        # Initialization
        self.labels = labels
        self.examples = examples
        self.transform = transform
        self.image_dir = args['image_dir']

    def __len__(self):
        # Denotes the total number of samples
        return len(self.examples)

    def __getitem__(self, idx):
        # Generates one sample of data
        # Convert to RGB to avoid png.
        id = self.examples[idx].strip()
        X = Image.open(self.image_dir + id).convert('RGB')
        X = self.transform(X)
        label = int(self.labels[idx])
        return X, label
