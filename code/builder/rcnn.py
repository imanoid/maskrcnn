import tensorflow as tf
import data.base as base


class RCNNBuilder(base.GraphBuilder):
    def __init__(self, dtype=tf.float32):
        base.GraphBuilder.__init__(self, dtype)

    def build_roi_detector(self, trunk_output, anchors):
        pass

    def build_roi_loss(self, roi_output, roi_grountruth):
        pass

    def build_obj_detector(self, trunk_output, roi_output, is_training):