import abc
import typing


class DataLoader(object):
    @abc.abstractmethod
    def initialize_data(self):
        pass

    @abc.abstractmethod
    def initialize_datasets(self,
                            datasets: typing.List[typing.Dict]):
        """
        :param datasets: list of dataset dicts (e.g. [{"name": "validation", "size": 25}])
        :return: None
        """
        pass

    @abc.abstractmethod
    def get_data(self,
                 dataset: str,
                 samples: int=None,
                 shuffle: bool=True,
                 features: typing.List=None) -> typing.List[typing.Dict]:
        """
        :param dataset: name of dataset
        :param samples: number of samples, None if all
        :param shuffle: true if dataset should be shuffled
        :param features: list of features the samples should have (i.e. bboxes, class, masks)
        :return: List of samples
        """
        pass


class ObjectInstance(object):
    def __init__(self,
                 label: str,
                 bounding_box: typing.Tuple[float, float, float, float]=None,
                 mask=None,
                 is_truncated: bool=False,
                 is_difficult: bool=False,
                 objectness: float=1):
        self.label = label
        self.bounding_box = bounding_box
        self.mask = mask
        self.is_truncated = is_truncated
        self.is_difficult = is_difficult