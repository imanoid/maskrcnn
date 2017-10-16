import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import data

if __name__ == "__main__":
    fname = "/media/imanoid/Data/workspace/data/VOCdevkit/VOC2012/SegmentationObject/2010_002763.png"
    img = ndimage.imread(fname)
    data.coding.detect_blobs(img, np.array([[0, 0, 0], [224, 224, 192]]))
