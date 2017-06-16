import os
import data.base as base
import pandas as pd
import xml.etree.ElementTree as ET
from scipy import ndimage, misc

class PascalVocDataLoader(base.DataLoader):
    def __init__(self,
                 config_name,
                 voc_dir,
                 n_test_samples,
                 n_valid_samples,
                 labels=None,
                 classification=False,
                 detection=False,
                 instance_segmentation=False,
                 class_segmentation=False,
                 image_shape=(192, 192, 3)):
        self.datasets_dir = os.path.join(voc_dir, "ImageSets")
        self.classificationset_dir = os.path.join(self.datasets_dir, "Main")
        self.segmentationset_dir = os.path.join(self.datasets_dir, "Segmentation")
        self.annotations_dir = os.path.join(voc_dir, "Annotations")
        self.images_dir = os.path.join(voc_dir, "JPEGImages")
        self.pickle_dir = os.path.join(voc_dir, "Pickle", config_name)

        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples

        self.labels = labels

        self.classification = classification
        self.detection = detection
        self.instance_segmentation = instance_segmentation
        self.class_segmentation = class_segmentation

        self.image_shape = image_shape

    def _load_images_from_files(self, image_files):
        images = []
        for image_file in image_files:
            images.append(self._load_image_from_file(image_file))
        return images

    def _load_image_from_file(self, image_file):
        image_data = (ndimage.imread(os.path.join(self.images_dir, image_file + ".jpg")).astype(float) -
                      self.image_shape[2] / 2) / self.image_shape[2]

        # resize image
        image_data = misc.imresize(image_data, self.image_shape)

        return image_data

    def _load_object_boxes_from_files(self, image_files):
        objects_boxes = []
        for image_file in image_files:
            objects_boxes.append(self._load_object_boxes_from_file(image_file))
        return objects_boxes

    def _load_object_boxes_from_file(self, image_file):
        ann = self._load_annotations_from_file(image_file)
        size = ann.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        object_boxes = []
        objs = ann.findall("object")
        for obj in objs:
            label = obj.find("name").text
            truncated = bool(obj.find("truncated").text)
            difficult = bool(obj.find("difficult").text)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            if self.labels is None or label in self.labels:
                object_boxes.append(base.BoundedObject(label,
                                                     (int(xmin / width * self.image_shape[0]),
                                                      int(ymin / height * self.image_shape[1]),
                                                      int(xmax / width * self.image_shape[0]),
                                                      int(ymax / height * self.image_shape[1])),
                                                     truncated,
                                                     difficult))
        return object_boxes

    def _load_annotations_from_file(self, image_file):
        annotations_file = os.path.join(self.annotations_dir, image_file) + '.xml'
        xml = ""
        with open(annotations_file, "r") as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        root = ET.fromstring(xml)
        return root

    def _load_class_mask_from_file(self, image_file):
        pass

    def _load_instance_masks_from_file(self, image_file, annotations):
        pass

    def _get_classification_image_files(self):
        filename = os.path.join(self.classificationset_dir, "trainval.txt")
        classification_files = []
        with open(filename, "r") as f:
            classification_files = [line[:-1] for line in f]
        return classification_files[:20]

    def _get_segmentation_image_files(self):
        filename = os.path.join(self.segmentationset_dir, "trainval.txt")
        with open(filename, "rb") as f:
            segmentation_files = f.readlines()
        return segmentation_files

    def _get_image_files_from_label(self, label):
        filename = os.path.join(self.classificationset_dir, label + "_trainval.txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        df = df[df['true'] == 1]
        return list(df['filename'].values)

    def _get_labels(self):
        all_files = os.listdir(self.classificationset_dir)
        labels = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0] for filename in all_files])))
        print(labels)
        labels.remove("trainval")
        labels.remove("val")

        return labels

    def initialize(self, reset=False):
        if self.detection:
            image_files = self._get_classification_image_files()
            bounding_boxes = self._load_object_boxes_from_files(image_files)
            return image_files, bounding_boxes

    def load_labels(self):
        if self.labels is not None:
            return self.labels
        else:
            return self._get_labels()

    def load_testset(self):
        pass

    def load_validset(self):
        pass

    def load_trainset(self):
        pass

    def load_trainset_random_minibatch(self, batch_size):
        pass