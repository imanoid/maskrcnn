import abc

class DataLoader(object):
    def __init__(self,
                 n_test_samples,
                 n_valid_samples,
                 labels=None,
                 classification=False,
                 detection=False,
                 instance_segmentation=False,
                 class_segmentation=False):
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples
        self.labels = labels
        self.classification = classification
        self.detection = detection
        self.instance_segmentation = instance_segmentation
        self.class_segmentation = class_segmentation

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