import data
import typing
import util
import numpy as np


def multiclass_batch(samples: typing.List[typing.Dict]) -> typing.Tuple[typing.Any, typing.Any]:
    n_samples = len(samples)

    first_sample = samples[0]
    first_image = first_sample["image"]
    first_labels = first_sample["multiclass_onehot"]

    images = np.zeros((n_samples, *first_image.shape), np.float32)
    labels = np.zeros((n_samples, len(first_labels)), np.float32)

    for i_sample, sample in enumerate(samples):
        images[i_sample, :, :, :] = sample["image"]
        labels[i_sample, :] = sample["multiclass_onehot"]

    return images, labels


class MinibatchLoader(object):
    def __init__(self,
                 data_loader: data.base.DataLoader,
                 sample_names, minibatch_size: int=20,
                 minibatch_factory=multiclass_batch):
        self.data_loader = data_loader
        self.sample_names = sample_names
        self.minibatch_size = minibatch_size
        self.minibatch_factory = minibatch_factory

    def random_minibatch(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        samples = self.data_loader.load_samples(util.random_subset(self.sample_names, self.minibatch_size))
        return self.minibatch_factory(samples)

    def sequential_minibatches(self):
        n_samples = len(self.sample_names)

        i_start = 0

        while i_start < n_samples:
            i_end = min(i_start + self.minibatch_size, n_samples)

            minibatch_samples = (self.data_loader.load_samples(self.sample_names[i_start:i_end]))
            yield self.minibatch_factory(minibatch_samples)

            i_start = i_end + 1
