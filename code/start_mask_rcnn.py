from data.pascal_voc import PascalVocSegmentationLoader
import os

if __name__ == "__main__":
    input_resolution = (192, 192)
    voc_path = "/media/imanoid/DATA/workspace/data/VOCdevkit/VOC2012"
    img_dir = os.path.join(voc_path, 'JPEGImages')
    ann_dir = os.path.join(voc_path, 'Annotations')
    set_dir = os.path.join(voc_path, 'ImageSets', 'Main')
    pickle_dir = os.path.join(voc_path, 'Pickle')

    loader = PascalVocSegmentationLoader(set_dir,
                                         ann_dir,
                                         img_dir,
                                         pickle_dir,
                                         image_resolution=input_resolution,
                                         n_valid_samples=20,
                                         n_test_samples=20)
    loader.initialize()