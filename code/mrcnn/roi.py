import tensorflow as tf

def roi_detector(featuremaps, n_classes, n_aspectratios):
    roi_outputs = []
    for featuremap in featuremaps:
        # add 3x3->256 conv
        # objectness + bbox: add 1x1->2+4*n_aspectratios conv
        roi_output = None
        roi_outputs.append(roi_output)
    return roi_outputs