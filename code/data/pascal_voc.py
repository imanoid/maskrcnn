import os
import numpy as np
import cPickle
import pandas as pd
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

import random
from base import DataLoader

class PascalVocLoader(object):
    def get_labels(self):
        all_files = os.listdir(self.set_dir)
        labels = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0] for filename in all_files])))

        labels.remove("train")
        labels.remove("trainval")
        labels.remove("val")

        return labels

    def _list_files_from_label(self, label, dataset):
        filename = os.path.join(self.set_dir, label + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        df = df[df['true'] == 1]
        return list(df['filename'].values)

    def _annotation_file_from_img(self, img_name):
        return os.path.join(self.ann_dir, img_name) + '.xml'

    def _load_annotation(self, img_filename):
        xml = ""
        with open(self._annotation_file_from_img(img_filename)) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        return ET.fromstring(xml)

    def _get_all_obj_and_box(self, img_list):
        for img in img_list:
            annotation = self._load_annotation(img)

    def _get_all_image_names(self, dataset):
        labels = self.get_labels()
        img_list = []
        for label in labels:
            label_files = self._list_files_from_label(label, dataset)
            for label_file in label_files:
                if label_file not in img_list:
                    img_list.append(label_file)
        return img_list

    def _load_image(self, image_file):
        # load normalised image
        image_data = (ndimage.imread(os.path.join(self.img_dir, image_file + ".jpg")).astype(float) -
                      self.pixel_depth / 2) / self.pixel_depth

        # resize image
        image_data = misc.imresize(image_data, (self.image_shape[0], self.image_shape[1], self.color_depth))

        return image_data

    def _load_images(self, image_files):
        images = np.ndarray(
            shape=(len(image_files), self.image_shape[0], self.image_shape[1], self.color_depth), dtype=np.float32)
        i_entry = 0
        for image_file in image_files:
            images[i_entry, :, :, :] = self._load_image(image_file)
            i_entry += 1

        return images

class PascalVocClassificationLoader(DataLoader, PascalVocLoader):
    def __init__(self,
                 set_dir,
                 ann_dir,
                 img_dir,
                 pickle_dir,
                 test_per_label=5,
                 valid_per_label=5,
                 minibatch_size=25,
                 image_resolution=(300, 300),
                 pixel_depth=255.0,
                 color_depth=3,
                 single_label="person"):
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

        self.single_label = single_label

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

    def show_images_of_label(self, label):
        labels = self.get_labels()

        total_images = 0
        # get all image file names from this label
        label_files = self._list_files_from_label(label, "trainval")
        random.shuffle(label_files)
        for label_file in label_files:
            total_images += len(label_files)

            img = self._load_image(label_file)
            plt.imshow(img)
            plt.show()

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

            if self.single_label is None:
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
            else:
                # get all image file names from this label
                positive_files = self._list_files_from_label(self.single_label, "trainval")
                random.shuffle(positive_files)

                negative_files = []
                for label in labels:
                    if label != self.single_label:
                        # get all image file names from this label
                        label_files = self._list_files_from_label(label, "trainval")

                        for label_file in label_files:
                            if label_file not in positive_files:
                                negative_files.append(label_file)

                while len(positive_files) < len(negative_files):
                    num_pos = len(positive_files)
                    num_neg = len(negative_files)

                    if num_neg - num_pos >= num_pos:
                        new_files = positive_files[:]
                    else:
                        new_files = positive_files[0:num_neg - num_pos]
                    random.shuffle(new_files)
                    positive_files += new_files

                random.shuffle(positive_files)
                random.shuffle(negative_files)

                # split images into test, valid and train dataset
                test_files += positive_files[0:self.test_per_label]
                valid_files += positive_files[self.test_per_label:self.test_per_label + self.valid_per_label]
                train_files += positive_files[self.test_per_label + self.valid_per_label:]

                # assign correct labels
                test_labels += [1 for i in range(self.test_per_label)]
                valid_labels += [1 for i in range(self.valid_per_label)]
                train_labels += [1 for i in range(len(positive_files) - self.test_per_label - self.valid_per_label)]

                test_files += negative_files[0:self.test_per_label]
                valid_files += negative_files[self.test_per_label:self.test_per_label + self.valid_per_label]
                train_files += negative_files[self.test_per_label + self.valid_per_label:]

                # assign correct labels
                test_labels += [0 for i in range(self.test_per_label)]
                valid_labels += [0 for i in range(self.valid_per_label)]
                train_labels += [0 for i in range(len(negative_files) - self.test_per_label - self.valid_per_label)]

                num_labels = 2

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

    def _make_onehot(self, labels, num_labels):
        return (np.arange(num_labels) == labels[:,None]).astype(np.float32)



class PascalVocSegmentationLoader(DataLoader, PascalVocLoader):
    def __init__(self,
                 set_dir,
                 ann_dir,
                 img_dir,
                 pickle_dir,
                 n_test_samples=5,
                 n_valid_samples=5,
                 minibatch_size=25,
                 image_resolution=(300, 300),
                 pixel_depth=255.0,
                 color_depth=3):
        self.set_dir = set_dir
        self.ann_dir = ann_dir
        self.img_dir = img_dir
        self.pickle_dir = pickle_dir

        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples

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

    def initialize(self, reset=False):
        all_files = self._get_all_image_names("trainval")
        random.shuffle(all_files)

        n_samples = len(all_files)

        test_start = 0
        test_end = self.n_test_samples
        valid_start = test_end
        valid_end = valid_start + self.n_valid_samples
        train_start = valid_end
        train_end = n_samples

        train_files = all_files[train_start:train_end]
        valid_files = all_files[valid_start:valid_end]
        test_files = all_files[test_start:test_end]

        for train_file in train_files:
            image = self._load_image(train_file)
            annotation = self._load_annotation(train_file)

            plt.imshow(image)
            plt.show()