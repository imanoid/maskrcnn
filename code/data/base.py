import abc
import typing
import util
import numpy as np


class DataLoader(object):
    @abc.abstractmethod
    def initialize_data(self):
        pass

    @abc.abstractmethod
    def load_samples(self, sample_names) -> typing.List[typing.Dict]:
        pass


class MulticlassMinibatchLoader(object):
    def __init__(self, data_loader: DataLoader, sample_names, minibatch_size: int=20):
        self.data_loader = data_loader
        self.sample_names = sample_names
        self.minibatch_size = minibatch_size

    def random_minibatch(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        minibatch = self.data_loader.load_samples(util.random_subset(self.sample_names, self.minibatch_size))
        first_sample = minibatch[0]
        first_image = first_sample["image"]
        first_labels = first_sample["labels"]
        images = np.zeros((self.minibatch_size, *first_image.shape), np.float32)
        labels = np.zeros((self.minibatch_size, *first_labels.shape), np.float32)
        return images, labels


class ObjectInstance(object):
    def __init__(self,
                 label: str,
                 bounding_box: typing.Tuple[float, float, float, float]=None,
                 mask=None,
                 is_truncated: bool=False,
                 is_difficult: bool=False):
        self.label = label
        self.bounding_box = bounding_box
        self.mask = mask
        self.is_truncated = is_truncated
        self.is_difficult = is_difficult


class ROIBox(object):
    def __init__(self,
                 objectness: float,
                 bounding_box: typing.Tuple[float, float, float, float]=None):
        self.objectness = objectness
        self.bounding_box = bounding_box
