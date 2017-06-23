import numpy as np

def encode_rcnn_roi_mask(input_shape, object_boxes, output_shape, anchors):
    """
    For each position and each anchor, we assign a positive label if: the anchor has highest IoU with a gt box or anchor has IoU > 0.7 with any gt box
                                                 a negative label if: the anchor has IoU < 0.3 for all gt boxes

    The bounding box of an anchor corresponds to the object with highest IoU (if positive label)

    :param input_shape: array (width, height, depth)
    :param object_boxes: list of tuples containing (xmin, ymin, xmax, ymax)
    :param output_shape: array (width, height)
    :param anchors: list of tuples containing (width, height)
    :return: ground truth output (width, height, (4+2) * len(anchors)), loss contribution mask (width, height)
    """
    output_depth = (4+2) * len(anchors) # for each anchor: bbox (4) + objectness (2)

    np.array(output_shape[0], output_shape[1], output_depth)

def encode_rcnn_bbox(anchor_bbox, roi_bbox):
    """

    :param anchor_bbox: tuple containing (xmin, ymin, xmax, ymax)
    :param roi_bbox: tuple containing (xmin, ymin, xmax, ymax)
    :return: Target Values -> tuple containing (tx, ty, tw, th)
    """
    x = roi_bbox[0]
    y = roi_bbox[1]
    w = roi_bbox[2] - x
    h = roi_bbox[3] - y

    xa = anchor_bbox[0]
    ya = anchor_bbox[1]
    wa = anchor_bbox[2] - xa
    ha = anchor_bbox[3] - ya

    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)

    return (tx, ty, tw, th)

def decode_rcnn_bbox(anchor_bbox, target_bbox):
    """
    :param anchor_bbox: tuple containing (xmin, ymin, xmax, ymax)
    :param target_bbox: tuple containing (tx, ty, tw, th)
    :return: ROI bbox -> tuple containing (xmin, ymin, xmax, ymax)
    """
    tx = target_bbox[0]
    ty = target_bbox[1]
    tw = target_bbox[2] - tx
    th = target_bbox[3] - ty

    xa = anchor_bbox[0]
    ya = anchor_bbox[1]
    wa = anchor_bbox[2] - xa
    ha = anchor_bbox[3] - ya

    x = tx * wa + xa
    y = ty * ha + ya
    w = np.exp(tw) * wa
    h = np.exp(th) * ha

    return (x, y, x + w, y + h)

def intersection_over_union(boxA, boxB):
    """
    :param boxA: tuple containing (xmin, ymin, xmax, ymax)
    :param boxB: tuple containing (xmin, ymin, xmax, ymax)
    :return:
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou