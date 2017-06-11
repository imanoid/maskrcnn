import os
import base
import pandas as pd
import xml.etree.ElementTree as ET

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
                 class_segmentation=False):
        self.datasets_dir = os.path.join(voc_dir, "ImageSets")
        self.classificationset_dir = os.path.join(self.datasets_dir, "Main")
        self.segmentationset_dir = os.path.join(self.datasets_dir, "Segmentation")
        self.annotations_dir = os.path.join(voc_dir, "Annotations")
        self.images_dir = os.path.join(voc_dir, "JPEGImages")
        self.pickle_dir = os.path.join(voc_dir, config_name)

        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples

        self.labels = labels

        self.classification = classification
        self.detection = detection
        self.instance_segmentation = instance_segmentation
        self.class_segmentation = class_segmentation

    def _load_images_from_files(self, image_files):
        pass

    def _load_image_from_file(self, image_file):
        pass

    def _load_object_boxes_from_files(self, image_files):
        pass

    def _load_object_boxes_from_file(self, image_file):
        pass

    def _load_labels_from_files(self, image_files):
        pass

    def _load_label_from_file(self, image_file):
        pass

    def _load_annotations_from_file(self, image_file):
        annotations_file = os.path.join(self.annotations_dir, image_file) + '.xml'
        xml = ""
        with open(self._annotation_file_from_img(annotations_file)) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        root = ET.fromstring(xml)

    def _load_class_mask(self, image_file):
        pass

    def _load_instance_masks(self, image_file, annotations):
        pass

    def _get_classification_image_files(self):
        filename = os.path.join(self.classificationset_dir, "trainval.txt")
        with open(filename, "rb") as f:
            classification_files = f.readlines()
        return classification_files

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
        all_files = os.listdir(self.datasets_dir)
        labels = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0] for filename in all_files])))

        labels.remove("train")
        labels.remove("trainval")
        labels.remove("val")

        return labels

    def initialize(self, reset=False):
        pass

    def load_testset(self):
        pass

    def load_validset(self):
        pass

    def load_trainset(self):
        pass

    def load_trainset_random_minibatch(self, batch_size):
        pass