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


class ObjectInstance(object):
    def __init__(self,
                 label: str,
                 bounding_box: typing.List[float]=None,
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
                 bounding_box: typing.List[float]=None):
        self.objectness = objectness
        self.bounding_box = bounding_box
