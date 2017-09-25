import typing
import os
import data.base as base
import data.coding as coding
import pandas as pd
import pickle
import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage, misc


class PascalVocDataLoader(base.DataLoader):
    def __init__(self,
                 config_name: str,
                 voc_dir: str,
                 image_shape: typing.List[float, float]=(256, 256),
                 rpn_shape: typing.List[float, float]=None,
                 anchors: typing.List[typing.List[float, float]]=None):
        self.datasets_dir = os.path.join(voc_dir, "ImageSets")
        self.classificationset_dir = os.path.join(self.datasets_dir, "Main")
        self.segmentationset_dir = os.path.join(self.datasets_dir, "Segmentation")
        self.annotations_dir = os.path.join(voc_dir, "Annotations")
        self.images_dir = os.path.join(voc_dir, "JPEGImages")
        self.pickle_dir = os.path.join(voc_dir, config_name)
        os.makedirs(self.pickle_dir)
        self.image_shape = image_shape  # (height, width)
        self.rpn_shape = rpn_shape,
        self.anchors = anchors

    def initialize_data(self):
        # init samples
        sample_names = []
        for file_name in os.listdir(self.images_dir):
            sample_name = file_name.replace(".jpg", "")
            sample_names.append(sample_name)

            # init sample
            sample = dict()

            # init image
            sample["image"] = self._load_image_from_sample(sample_name)

            # init bounding boxes and masks
            object_instances = self._load_objects_from_sample(sample_name)
            sample["objects"] = object_instances

            # init RPN output
            if self.rpn_shape is None:
                print("No rpn shape specified. Not initializing rpn output!")
            elif self.anchors is None:
                print("No anchors specified. Not initializting rpn output!")
            else:
                rpn_output, rpn_loss_mask = coding.encode_rpn_output(self.image_shape,
                                                                     coding.objects_to_bboxes(object_instances),
                                                                     self.rpn_shape,
                                                                     self.anchors)
                sample["rpn_output"] = rpn_output
                sample["rpn_loss_mask"] = rpn_loss_mask

            #"objects": None,
                      #"segmentation": None,
                      #"labels": None}
            self._save_sample(sample, sample_name)

        # init sample labels
        labels = self._load_labels()
        for label in labels:
            label_samples = self._load_image_files_from_label(label)
            for label_sample in label_samples:
                sample = self._load_sample(label_sample)
                if "labels" in sample:
                    sample["labels"] = None
                sample["labels"].append(label)
                self._save_sample(sample, label_sample)

    # samples
    def _load_sample(self, sample_name: typing.AnyStr) -> typing.Dict:
        with open(os.path.join(self.pickle_dir, "{}.json".format(sample_name)), "rb") as f:
            return pickle.load(f)

    def _save_sample(self, sample: typing.Dict, sample_name: typing.AnyStr):
        with open(os.path.join(self.pickle_dir, "{}.json".format(sample_name)), "wb") as f:
            pickle.dump(sample, f)

    # images
    def _load_image_from_sample(self, sample_name: typing.AnyStr) -> np.ndarray:
        # read image
        image_data = ndimage.imread(os.path.join(self.images_dir, sample_name + ".jpg")).astype(np.float32)

        # resize image
        image_data = misc.imresize(image_data, self.image_shape)

        return image_data

    # objects
    def _load_objects_from_sample(self, sample_name: typing.AnyStr) -> typing.List[base.ObjectInstance]:
        ann = self._load_annotations_from_file(sample_name)
        size = ann.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        object_boxes = []

        # initialize bounding boxes
        objs = ann.findall("object")
        for obj in objs:
            label = obj.find("name").text
            truncated = bool(obj.find("truncated").text)
            difficult = bool(obj.find("difficult").text)
            bbox = obj.find("bndbox")
            ymin = float(bbox.find("ymin").text) / height
            xmin = float(bbox.find("xmin").text) / width
            ymax = float(bbox.find("ymax").text) / height
            xmax = float(bbox.find("xmax").text) / width

            if self.labels is None or label in self.labels:
                object_boxes.append(base.ObjectInstance(label,
                                                        bounding_box=[ymin, xmin, ymax, xmax],
                                                        is_truncated=truncated,
                                                        is_difficult=difficult))
        return object_boxes

    # annotations
    def _load_annotations_from_file(self, image_file):
        annotations_file = os.path.join(self.annotations_dir, "{}.xml".format(image_file))
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

    def _load_classification_image_files(self):
        filename = os.path.join(self.classificationset_dir, "trainval.txt")
        classification_files = []
        with open(filename, "r") as f:
            classification_files = [line[:-1] for line in f]
        return classification_files[:20]

    def _load_segmentation_image_files(self):
        filename = os.path.join(self.segmentationset_dir, "trainval.txt")
        with open(filename, "rb") as f:
            segmentation_files = f.readlines()
        return segmentation_files

    def _load_image_files_from_label(self, label):
        filename = os.path.join(self.classificationset_dir, label + "_trainval.txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        df = df[df['true'] == 1]
        return list(df['filename'].values)

    def _load_labels(self):
        all_files = os.listdir(self.classificationset_dir)
        labels = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0] for filename in all_files])))
        labels.remove("trainval")
        labels.remove("val")

        return labels