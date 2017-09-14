import numpy as np
import typing
import data.base as base


max_false_iou = 0.3
min_true_iou = 0.7


def objects_to_bboxes(object_instances: typing.List[base.ObjectInstance]) \
        -> typing.List[typing.List[float, float, float, float]]:
    """
    convert list of object instances to list of bounding boxes
    :param object_instances: list of object instances
    :return: list of bboxes
    """
    object_bboxes = list()
    for object_instance in object_instances:
        object_bboxes.append(object_instance.bounding_box)
    return object_bboxes


def image_coordinates(target_shape: typing.List[int, int],
                      coordinates: typing.List[float, float]) \
        -> typing.List[float, float]:
    """
    :param target_shape: target (height, width)
    :param coordinates: coordinates to transform (y, x)
    :return: transformed coordinates (y, x)
    """
    return [coordinates[0] * target_shape[0],
            coordinates[1] * target_shape[1]]


def normalize_coordinates(source_shape: typing.List[int, int],
                          coordinates: typing.List[float, float]) \
        -> typing.List[float, float]:
    """
    :param source_shape: target (height, width)
    :param coordinates: coordinates to transform (y, x)
    :return: normalized coordinates (y, x)
    """
    return [coordinates[0] / source_shape[0],
            coordinates[1] / source_shape[1]]


def translate_bbox(target_shape: typing.List[int, int],
                   box: typing.List[float, float, float, float]) \
        -> typing.List[float, float, float, float]:
    """
    :param target_shape: target (height, width)
    :param box: coordinates to transform (ymin, xmin, ymax, xmax)
    :return: transformed coordinates (ymin, xmin, ymax, xmax)
    """
    return [box[0] * target_shape[0],
            box[1] * target_shape[1],
            box[2] * target_shape[0],
            box[3] * target_shape[1]]


def encode_rpn_output(object_bboxes: typing.List[typing.List[float, float, float, float]],
                      output_shapes: typing.List[typing.List[int, int]],
                      anchors: typing.List[typing.List[float, float]]) \
        -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray], typing.List[np.ndarray]]:
    """
    For each position and each anchor, we assign a positive label if: the anchor has highest IoU with a gt box or anchor has IoU > 0.7 with any gt box
                                                 a negative label if: the anchor has IoU < 0.3 for all gt boxes

    The bounding box of an anchor corresponds to the object with highest IoU (if positive label)

    :param object_bboxes: list of tuples containing (ymin, xmin, ymax, xmax)
    :param output_shapes: list of tuples (height, width)
    :param anchors: list of tuples containing (height, width)
    :return: ground truth regression list of numpy arrays (height, width, 4),
             ground truth objectness list of numpy arrays (height, width, 2),
             loss contribution mask list of numpy arrays (height, width)
    """
    assert(len(anchors) == len(output_shapes))

    regression_outputs = list()
    objectness_outputs = list()
    loss_masks = list()

    for i_anchor in range(len(anchors)):
        output_shape = output_shapes[i_anchor]
        anchor = anchors[i_anchor]

        ious = np.zeros(output_shape, np.float32)

        regression_output = np.zeros([*output_shape, 4], np.float32)
        objectness_output = np.zeros([*output_shape, 2], np.float32)
        loss_mask = np.ones(output_shape, np.bool)

        for object_bbox in object_bboxes:
            for y_output_pos in range(output_shape[0]):
                for x_output_pos in range(output_shape[1]):
                    center = normalize_coordinates(output_shape, [y_output_pos, x_output_pos])
                    anchor_bbox_output = get_anchor_bbox(anchor, center)
                    iou = intersection_over_union(anchor_bbox_output, object_bbox)

                    if ious[y_output_pos, x_output_pos] < iou:
                        ious[y_output_pos, x_output_pos] = iou
                        target_bbox = encode_rpn_bbox(anchor_bbox_output, object_bbox)
                        regression_output[y_output_pos, x_output_pos, :] = target_bbox

        for y_output_pos in range(output_shape[0]):
            for x_output_pos in range(output_shape[1]):
                iou = ious[y_output_pos, x_output_pos]
                if iou < max_false_iou:
                    # false sample
                    objectness_output[y_output_pos, x_output_pos, :] = [0, 1]
                elif iou > min_true_iou:
                    # true sample
                    objectness_output[y_output_pos, x_output_pos, :] = [1, 0]
                else:
                    # ignore sample
                    loss_mask[y_output_pos, x_output_pos] = False

    return regression_outputs, objectness_outputs, loss_masks


def decode_rpn_output(objectness_output: np.ndarray,
                      regression_output: np.ndarray,
                      anchor: typing.List[float, float]) \
        -> typing.List[base.ROIBox]:
    """
    Decode the RPN output from the nn.

    :param objectness_output: numpy array with shape (height, width, 2)
    :param regression_output: numpy array with shape (height, width, 4)
    :param anchor: tuple containing (height, width)
    :return: list of roi bboxes
    """
    output_shape = objectness_output.shape[0:2]
    roi_bboxes = list()

    for y_output_pos in range(output_shape[0]):
        for x_output_pos in range(output_shape[1]):
            objectness = objectness_output[y_output_pos, x_output_pos, 0]
            if objectness > 0.5:
                center = normalize_coordinates(output_shape, [y_output_pos, x_output_pos])
                anchor_bbox = get_anchor_bbox(anchor, center)
                encoded_bbox = regression_output[y_output_pos, x_output_pos, :]
                decoded_bbox = decode_rpn_bbox(anchor_bbox, encoded_bbox)
                roi_box = base.ROIBox(objectness=objectness,
                                      bounding_box=decoded_bbox)
                roi_bboxes.append(roi_box)

    return roi_bboxes


def encode_anchor_bboxes(output_shape: typing.List[int, int],
                         anchor: typing.List[float, float]):
    """
    :param input_shape: tuple containing (height, width)
    :param output_shape: tuple containing (height, width)
    :param anchor: tuple containing (height, width)
    :return: numpy array of shape (height, width, 4) where the last dimension contains (ymin, xmin, ymax, xmax)
    """
    anchor_bboxes = np.zeros((*output_shape, 4), np.float32)

    for y_output_pos in range(output_shape[0]):
        for x_output_pos in range(output_shape[1]):
            center = normalize_coordinates(output_shape, [y_output_pos, x_output_pos])
            anchor_bbox = get_anchor_bbox(anchor, center)
            anchor_bbox[2] -= anchor_bbox[0]
            anchor_bbox[3] -= anchor_bbox[1]
            anchor_bboxes[y_output_pos, x_output_pos, :] = anchor_bbox

    return anchor_bboxes


def get_anchor_bbox(anchor: typing.List[float, float],
                    center: typing.List[float, float]) \
        -> typing.List[float, float, float, float]:
    ymin = center[0] - anchor[0] / 2
    xmin = center[1] - anchor[1] / 2
    ymax = ymin + anchor[0]
    xmax = xmin + anchor[1]

    return [ymin, xmin, ymax, xmax]


def encode_rpn_bbox(anchor_bbox: typing.List[float, float, float, float],
                    roi_bbox: typing.List[float, float, float, float]) \
        -> typing.List[float, float, float, float]:
    """

    :param anchor_bbox: tuple containing (ymin, xmin, ymax, xmax)
    :param roi_bbox: tuple containing (ymin, xmin, ymax, xmax)
    :return: Target Values -> tuple containing (tx, ty, tw, th)
    """
    y = roi_bbox[0]
    x = roi_bbox[1]
    h = roi_bbox[2] - y
    w = roi_bbox[3] - x

    ay = anchor_bbox[0]
    ax = anchor_bbox[1]
    ah = anchor_bbox[2] - ay
    aw = anchor_bbox[3] - ax

    ty = (y - ay) / ah
    tx = (x - ax) / aw
    th = np.log(h / ah)
    tw = np.log(w / aw)

    return [ty, tx, th, tw]


def decode_rpn_bbox(anchor_bbox: typing.List[float, float, float, float],
                    target_bbox: typing.List[float, float, float, float]) \
        -> typing.List[float, float, float, float]:
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

    return [y, x, y + h, x + w]


def intersection_over_union(box_a: typing.List[float, float, float, float],
                            box_b: typing.List[float, float, float, float]) \
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