from data import pascal, visualization
import os

if __name__ == "__main__":
    input_resolution = (192, 192)
    voc_path = "/media/imanoid/DATA/workspace/data/VOCdevkit/VOC2012"
    img_dir = os.path.join(voc_path, 'JPEGImages')
    ann_dir = os.path.join(voc_path, 'Annotations')
    set_dir = os.path.join(voc_path, 'ImageSets', 'Main')
    pickle_dir = os.path.join(voc_path, 'Pickle')

    loader = pascal.PascalVocDataLoader("DetectionAndClassification",
                 voc_path,
                 40,
                 40,
                 detection=True,
                 image_shape=(192, 192, 3))
    image_files, bboxes = loader.initialize()
    images = loader._load_images_from_files(image_files)

    visualizer = visualization.Visualizer(loader.load_labels())
    for i in range(len(images)):
        visualizer.show_objects(images[i], bboxes[i])