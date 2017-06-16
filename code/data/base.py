import abc

class DataLoader(object):
    @abc.abstractmethod
    def initialize(self, reset=False):
        pass

    @abc.abstractmethod
    def load_testset(self):
        pass

    @abc.abstractmethod
    def load_validset(self):
        pass

    @abc.abstractmethod
    def load_trainset(self):
        pass

    @abc.abstractmethod
    def load_trainset_random_minibatch(self, batch_size):
        pass


class BoundedObject(object):
    def __init__(self,
                 label,
                 bounding_box,
                 is_truncated,
                 is_difficult):
        self.label = label
        self.bounding_box = bounding_box
        self.is_truncated = is_truncated
        self.is_difficult = is_difficult