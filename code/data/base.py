import abc

class DataLoader(object):
    @abc.abstractmethod
    def initialize_data(self):
        pass

    @abc.abstractmethod
    def initialize_datasets(self, datasets):
        """
        :param datasets: list of dataset dicts (e.g. [{"name": "validation", "size": 25}])
        :return: None
        """
        pass

    @abc.abstractmethod
    def get_data(self, dataset, samples=None, shuffle=True, features=None):
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
                 label,
                 bounding_box=None,
                 mask=None,
                 is_truncated=False,
                 is_difficult=False):
        self.label = label
        self.bounding_box = bounding_box
        self.mask = mask
        self.is_truncated = is_truncated
        self.is_difficult = is_difficult