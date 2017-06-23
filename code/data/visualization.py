from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt

class Visualizer(object):
    def __init__(self,
                 labels):
        self.labels = labels
        self.colors = []

        for label in labels:
            color = tuple((np.random.rand(3) * 255).astype(np.int32).tolist())
            self.colors.append(color)

    def show_objects(self, img, object_boxes, masks=None):
        print(img.shape)
        pimage = Image.fromarray(img)

        draw = ImageDraw.Draw(pimage)
        for object_box in object_boxes:
            draw.rectangle(object_box.bounding_box, outline=self.colors[self.labels.index(object_box.label)])

        plt.imshow(pimage)
        plt.show()
