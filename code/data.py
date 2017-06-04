import os
import numpy as np
import cPickle
import pandas as pd
from bs4 import BeautifulSoup
import voc_utils
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import abc
import util

import random

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

class PascalVocLoader(object):
    def __init__(self, set_dir, ann_dir, img_dir, pickle_dir, test_per_label=2, valid_per_label=2, minibatch_size=25, image_resolution=(300, 300), pixel_depth=255.0, color_depth=3):
        self.set_dir = set_dir
        self.ann_dir = ann_dir
        self.img_dir = img_dir
        self.pickle_dir = pickle_dir

        self.test_per_label = test_per_label
        self.valid_per_label = valid_per_label

        self.minibatch_size = minibatch_size

        self.image_shape = image_resolution
        self.pixel_depth = pixel_depth
        self.color_depth = color_depth

        self.test_dataset = None
        self.valid_dataset = None

        self.test_labels = None
        self.valid_labels = None

        self.minibatch_files = None

        self.labels = None

    def show_image_per_label(self):
        labels = self.get_labels()

        total_images = 0
        for label in labels:
            # get all image file names from this label
            label_files = self._list_files_from_label(label, "trainval")
            total_images += len(label_files)
            print("%s: %d" % (label, len(label_files)))
            img = self._load_image(random.choice(label_files))
            plt.imshow(img)
            plt.show()
        print("total: %d" % total_images)

    def get_labels(self):
        all_files = os.listdir(self.set_dir)
        labels = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0] for filename in all_files])))

        labels.remove("train")
        labels.remove("trainval")
        labels.remove("val")

        return labels

    def initialize(self, reset=False):
        dataset_index_file = os.path.join(self.pickle_dir, "dataset.pickle")
        if os.path.exists(dataset_index_file) and not reset:
            dataset_index = cPickle.load(open(dataset_index_file, "rb"))

            self.test_dataset = dataset_index["test_dataset"]
            self.valid_dataset = dataset_index["valid_dataset"]

            self.test_labels = dataset_index["test_labels"]
            self.valid_labels = dataset_index["valid_labels"]

            self.minibatch_files = dataset_index["minibatch_files"]

            self.labels = dataset_index["labels"]
        else:
            # initialize classification
            labels = self.get_labels()

            self.labels = labels

            num_labels = len(labels)

            test_files = []
            valid_files = []
            train_files = []

            test_labels = []
            valid_labels = []
            train_labels = []

            i_label = 0
            for label in labels:
                # get all image file names from this label
                label_files = self._list_files_from_label(label, "trainval")
                random.shuffle(label_files)

                # split images into test, valid and train dataset
                test_files += label_files[0:self.test_per_label]
                valid_files += label_files[self.test_per_label:self.test_per_label+self.valid_per_label]
                train_files += label_files[self.test_per_label+self.valid_per_label:]

                # assign correct labels
                test_labels += [i_label for i in range(self.test_per_label)]
                valid_labels += [i_label for i in range(self.valid_per_label)]
                train_labels += [i_label for i in range(len(label_files) - self.test_per_label - self.valid_per_label)]


                i_label += 1

            self.test_dataset = self._load_images(test_files)
            self.valid_dataset = self._load_images(valid_files)

            self.test_labels = self._make_onehot(np.array(test_labels, dtype=np.int32), num_labels)
            self.valid_labels = self._make_onehot(np.array(valid_labels, dtype=np.int32), num_labels)

            self.minibatch_files = self._split_into_minibatches(train_files, self._make_onehot(np.array(train_labels, dtype=np.int32), num_labels))

            dataset_index = {
                "test_dataset": self.test_dataset,
                "valid_dataset": self.valid_dataset,

                "test_labels": self.test_labels,
                "valid_labels": self.valid_labels,

                "minibatch_files": self.minibatch_files,

                "labels": self.labels
            }

            cPickle.dump(dataset_index, open(dataset_index_file, "wb"))

    def load_testset(self):
        return self.test_dataset, self.test_labels

    def load_validset(self):
        return self.valid_dataset, self.valid_labels

    def load_trainset(self):
        # return self.train_dataset, self.train_labels
        raise NotImplementedError("load_trainset not implemented for PascalVocLoader")

    def load_trainset_random_minibatch(self, batch_size):
        minibatch_file = random.sample(self.minibatch_files, 1)
        batch_dataset, batch_labels = cPickle.load(open(minibatch_file[0], "rb"))

        if batch_dataset.shape[0] <= batch_size:
            return batch_dataset, batch_labels

        permutation = np.random.permutation(batch_dataset.shape[0])

        return batch_dataset[permutation[0:batch_size], :, :], batch_labels[permutation[0:batch_size]]

    def _split_into_minibatches(self, train_files, train_labels):
        n_samples = len(train_files)
        permutation = random.sample(range(n_samples), n_samples)

        train_files = [train_files[i] for i in permutation]
        train_labels = train_labels[permutation]

        minibatch_files = []

        num_minibatches = 0

        for i_minibatch in range(int(len(train_files)) / self.minibatch_size + 1):
            minibatch_start = i_minibatch * self.minibatch_size
            minibatch_end = min(minibatch_start + self.minibatch_size, len(train_files) - 1)

            if minibatch_start == minibatch_end:
                break

            minibatch_dataset = self._load_images(train_files[minibatch_start:minibatch_end])
            minibatch_labels = train_labels[minibatch_start:minibatch_end]

            minibatch_file = os.path.join(self.pickle_dir, "minibatch%d.pickle" % i_minibatch)

            cPickle.dump((minibatch_dataset, minibatch_labels), open(minibatch_file, "wb"))

            minibatch_files.append(minibatch_file)

            num_minibatches += 1
            print("Minibatch %d" % i_minibatch)

        return minibatch_files

    def _random_permutation(self, image_files, image_labels, size=None):
        if size is None:
            size = len(image_files)
        permutation = random.sample(range(len(image_files)), size)
        return [image_files[i] for i in permutation], image_labels[permutation]

    def _load_images(self, image_files):
        images = np.ndarray(
            shape=(len(image_files), self.image_shape[0], self.image_shape[1], self.color_depth), dtype=np.float32)
        i_entry = 0
        for image_file in image_files:
            images[i_entry, :, :, :] = self._load_image(image_file)
            i_entry += 1

        return images

    def _make_onehot(self, labels, num_labels):
        return (np.arange(num_labels) == labels[:,None]).astype(np.float32)

    def _list_files_from_label(self, label, dataset):
        filename = os.path.join(self.set_dir, label + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        df = df[df['true'] == 1]
        return list(df['filename'].values)

    def _load_image(self, image_file):
        # load normalised image
        image_data = (ndimage.imread(os.path.join(self.img_dir, image_file + ".jpg")).astype(float) -
                      self.pixel_depth / 2) / self.pixel_depth

        # resize image
        image_data = misc.imresize(image_data, (self.image_shape[0], self.image_shape[1], self.color_depth))

        return image_data