import numpy as np
import typing


def encode_rcnn_roi_mask(input_shape: typing.Tuple[int, int],
                         object_bboxes: typing.List[typing.Tuple[float, float, float, float]],
                         output_shape: typing.Tuple[int, int],
                         anchors: typing.List[typing.Tuple[float, float]]) -> np.ndarray:
    """
    For each position and each anchor, we assign a positive label if: the anchor has highest IoU with a gt box or anchor has IoU > 0.7 with any gt box
                                                 a negative label if: the anchor has IoU < 0.3 for all gt boxes

    The bounding box of an anchor corresponds to the object with highest IoU (if positive label)

    :param input_shape: array (width, height)
    :param object_bboxes: list of tuples containing (xmin, ymin, xmax, ymax)
    :param output_shape: array (width, height)
    :param anchors: list of tuples containing (width, height)
    :return: ground truth output (width, height, (4+2) * len(anchors)), loss contribution mask (width, height)
    """
    output_depth = (4+2) * len(anchors)  # for each anchor: bbox (4) + objectness (2)

    output = np.array([output_shape[0], output_shape[1], output_depth])
    loss_mask = np.array([output_shape[0], output_shape[1], len(anchors)])

    for object_bbox in object_bboxes:
        for i_anchor, anchor in enumerate(anchors):
            for x_input_pos in range(input_shape[0]):
                for y_input_pos in range(input_shape[1]):
                    center = (x_input_pos, y_input_pos)
                    anchor_bbox = get_anchor_bbox(anchor, center)
                    iou = intersection_over_union(anchor_bbox, object_bbox)
    return output


def get_anchor_bbox(anchor: typing.Tuple[float, float],
                    center: typing.Tuple[float, float]) -> typing.Tuple[float, float, float, float]:
    xmin = center[0] - anchor[0] / 2
    ymin = center[1] - anchor[1] / 2
    xmax = xmin + anchor[0]
    ymax = ymin + anchor[1]

    return xmin, ymin, xmax, ymax


def encode_rcnn_bbox(anchor_bbox: typing.Tuple[float, float, float, float],
                     roi_bbox: typing.Tuple[float, float, float, float]) -> typing.Tuple[float, float, float, float]:
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

    return tx, ty, tw, th


def decode_rcnn_bbox(anchor_bbox: typing.Tuple[float, float, float, float],
                     target_bbox: typing.Tuple[float, float, float, float]) -> typing.Tuple[float, float, float, float]:
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

    return x, y, x + w, y + h


def intersection_over_union(box_a: typing.Tuple[float, float, float, float],
                            box_b: typing.Tuple[float, float, float, float]) -> float:
    """
    :param box_a: tuple containing (xmin, ymin, xmax, ymax)
    :param box_b: tuple containing (xmin, ymin, xmax, ymax)
    :return: iou
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou