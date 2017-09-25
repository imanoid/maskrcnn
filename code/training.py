import numpy as np


def random_minibatch(dataset, labels, batch_size):
    permutation = np.random.permutation(dataset.shape[0])
    return dataset[permutation[0:batch_size], :, :], labels[permutation[0:batch_size]]