import numpy as np
import typing
import data.base as base


max_false_iou = 0.3
min_true_iou = 0.7


def objects_to_bboxes(object_instances: typing.List[base.ObjectInstance]) \
        -> typing.Tuple[float, float, float, float]:
    """
    convert list of object instances to list of bounding boxes
    :param object_instances: list of object instances
    :return: list of bboxes
    """
    object_bboxes = list()
    for object_instance in object_instances:
        object_bboxes.append(object_instance.bounding_box)
    return object_bboxes


def encode_rpn_output(input_shape: typing.Tuple[int, int],
                      object_bboxes: typing.List[typing.Tuple[float, float, float, float]],
                      output_shape: typing.Tuple[int, int],
                      anchors: typing.List[typing.Tuple[float, float]]) \
        -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    For each position and each anchor, we assign a positive label if: the anchor has highest IoU with a gt box or anchor has IoU > 0.7 with any gt box
                                                 a negative label if: the anchor has IoU < 0.3 for all gt boxes

    The bounding box of an anchor corresponds to the object with highest IoU (if positive label)

    :param input_shape: array (height, width)
    :param object_bboxes: list of tuples containing (xmin, ymin, xmax, ymax)
    :param output_shape: array (height, width)
    :param anchors: list of tuples containing (height, width)
    :return: ground truth output (width, height, (4+2) * len(anchors)), loss contribution mask (height, width)
    """
    n_anchors = len(anchors)

    output = np.zeros([output_shape[0], output_shape[1], 6 * n_anchors], np.float32)  # for each anchor: objectness (2) + bbox (4)
    loss_mask = np.zeros([output_shape[0], output_shape[1], n_anchors], np.bool)

    for object_bbox in object_bboxes:
        for i_anchor, anchor in enumerate(anchors):
            iou_pos = i_anchor * 6
            target_start = iou_pos + 2
            target_end = iou_pos + 6
            for y_input_pos in range(input_shape[0]):
                for x_input_pos in range(input_shape[1]):
                    center = (y_input_pos, x_input_pos)
                    anchor_bbox = get_anchor_bbox(anchor, center)
                    iou = intersection_over_union(anchor_bbox, object_bbox)
                    if output[y_input_pos, x_input_pos, iou_pos] < iou:
                        output[y_input_pos, x_input_pos, iou_pos] = iou
                        target_bbox = encode_rpn_bbox(anchor_bbox, object_bbox)
                        output[y_input_pos, x_input_pos, target_start:target_end] = target_bbox

    for i_anchor in range(n_anchors):
        for y_input_pos in range(input_shape[0]):
            for x_input_pos in range(input_shape[1]):
                output_start = i_anchor * 6

                if output[y_input_pos, x_input_pos, i_anchor * 6] < min_true_iou:
                    # false sample
                    output[y_input_pos, x_input_pos, output_start: output_start + 2] = (0, 1)
                    loss_mask[x_input_pos, x_input_pos, i_anchor] = True
                elif output[y_input_pos, x_input_pos, i_anchor * 6] > max_false_iou:
                    # true sample
                    loss_mask[y_input_pos, x_input_pos, i_anchor] = True
                    output[y_input_pos, x_input_pos, output_start: output_start + 2] = (1, 0)
                else:
                    # ignore sample
                    output[y_input_pos, x_input_pos, output_start: output_start + 2] = (0, 0)
                    loss_mask[y_input_pos, x_input_pos, i_anchor] = False

    return output, loss_mask


def decode_rpn_output(output: np.ndarray,
                      anchors: typing.List[typing.Tuple[float, float]]) \
        -> typing.List[typing.Tuple[float, float, float, float]]:
    """
    Decode the RPN output from the nn.

    :param output: the output
    :param anchors: the anchors
    :return: list of roi bboxes
    """
    output_shape = output.shape
    roi_bboxes = list()

    for i_anchor, anchor in enumerate(anchors):
        for y_output_pos in range(output_shape[0]):
            for x_output_pos in range(output_shape[1]):
                output_start = i_anchor * 6
                output_bbox_start = output_start + 2
                output_bbox_end = output_bbox_start + 4
                objectness = output[y_output_pos, x_output_pos, output_start:output_bbox_start]
                if np.argmax(objectness) == 0:
                    encoded_bbox = output[y_output_pos, x_output_pos, output_bbox_start:output_bbox_end]
                    decoded_bbox = decode_rpn_bbox(anchor, encoded_bbox)
                    roi_bboxes.append(decoded_bbox)

    return roi_bboxes


def get_anchor_bbox(anchor: typing.Tuple[float, float],
                    center: typing.Tuple[float, float]) \
        -> typing.Tuple[float, float, float, float]:
    ymin = center[0] - anchor[0] / 2
    xmin = center[1] - anchor[1] / 2
    ymax = ymin + anchor[0]
    xmax = xmin + anchor[1]

    return ymin, xmin, ymax, xmax


def encode_rpn_bbox(anchor_bbox: typing.Tuple[float, float, float, float],
                    roi_bbox: typing.Tuple[float, float, float, float]) \
        -> typing.Tuple[float, float, float, float]:
    """

    :param anchor_bbox: tuple containing (xmin, ymin, xmax, ymax)
    :param roi_bbox: tuple containing (xmin, ymin, xmax, ymax)
    :return: Target Values -> tuple containing (tx, ty, tw, th)
    """
    y = roi_bbox[0]
    x = roi_bbox[1]
    h = roi_bbox[2] - y
    w = roi_bbox[3] - x

    ya = anchor_bbox[0]
    xa = anchor_bbox[1]
    ha = anchor_bbox[2] - ya
    wa = anchor_bbox[3] - xa

    ty = (y - ya) / ha
    tx = (x - xa) / wa
    th = np.log(h / ha)
    tw = np.log(w / wa)

    return ty, tx, th, tw


def decode_rpn_bbox(anchor_bbox: typing.Tuple[float, float, float, float],
                    target_bbox: typing.Tuple[float, float, float, float]) \
        -> typing.Tuple[float, float, float, float]:
    """
    :param anchor_bbox: tuple containing (ymin, xmin, ymax, xmax)
    :param target_bbox: tuple containing (ty, tx, th, tw)
    :return: ROI bbox -> tuple containing (ymin, xmin, ymax, xmax)
    """
    ty = target_bbox[0]
    tx = target_bbox[1]
    th = target_bbox[2] - ty
    tw = target_bbox[3] - tx

    ay = anchor_bbox[0]
    ax = anchor_bbox[1]
    ah = anchor_bbox[2] - ay
    aw = anchor_bbox[3] - ax

    y = ty * ah + ay
    x = tx * aw + ax
    h = np.exp(th) * ah
    w = np.exp(tw) * aw

    return y, x, y + h, x + w


def intersection_over_union(box_a: typing.Tuple[float, float, float, float],
                            box_b: typing.Tuple[float, float, float, float]) \
        -> float:
    """
    :param box_a: tuple containing (ymin, xmin, ymax, xmax)
    :param box_b: tuple containing (ymin, xmin, ymax, xmax)
    :return: iou
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = max(box_a[0], box_b[0])
    xA = max(box_a[1], box_b[1])
    yB = min(box_a[2], box_b[2])
    xB = min(box_a[3], box_b[3])

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