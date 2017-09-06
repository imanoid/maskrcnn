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


def translate_coordinates(source_shape: typing.Tuple[int, int],
                          target_shape: typing.Tuple[int, int],
                          coordinates: typing.Tuple[float, float]) \
        -> typing.Tuple[float, float]:
    """
    :param source_shape: source (height, width)
    :param target_shape: target (height, width)
    :param coordinates: coordinates to transform (y, x)
    :return: transformed coordinates (y, x)
    """
    return (coordinates[0] / source_shape[0] * target_shape[0],
            coordinates[1] / source_shape[1] * target_shape[1])


def translate_bbox(source_shape: typing.Tuple[int, int],
                   target_shape: typing.Tuple[int, int],
                   box: typing.Tuple[float, float, float, float]) \
        -> typing.Tuple[float, float, float, float]:
    """
    :param source_shape: source (height, width)
    :param target_shape: target (height, width)
    :param box: coordinates to transform (ymin, xmin, ymax, xmax)
    :return: transformed coordinates (ymin, xmin, ymax, xmax)
    """
    return (box[0] / source_shape[0] * target_shape[0],
            box[1] / source_shape[1] * target_shape[1],
            box[2] / source_shape[0] * target_shape[0],
            box[3] / source_shape[1] * target_shape[1])


def encode_rpn_output(input_shape: typing.Tuple[int, int],
                      object_bboxes: typing.List[typing.Tuple[float, float, float, float]],
                      output_shapes: typing.List[typing.Tuple[int, int]],
                      anchors: typing.List[typing.Tuple[float, float]]) \
        -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray], typing.List[np.ndarray]]:
    """
    For each position and each anchor, we assign a positive label if: the anchor has highest IoU with a gt box or anchor has IoU > 0.7 with any gt box
                                                 a negative label if: the anchor has IoU < 0.3 for all gt boxes

    The bounding box of an anchor corresponds to the object with highest IoU (if positive label)

    :param input_shape: array (height, width)
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
                    object_bbox_output = translate_bbox(input_shape,
                                                        output_shape,
                                                        object_bbox)
                    anchor_output = translate_coordinates(input_shape,
                                                          output_shape,
                                                          anchor)
                    center = (y_output_pos, x_output_pos)
                    anchor_bbox_output = get_anchor_bbox(anchor_output, center)
                    iou = intersection_over_union(anchor_bbox_output, object_bbox_output)

                    if ious[y_output_pos, x_output_pos] < iou:
                        ious[y_output_pos, x_output_pos] = iou
                        target_bbox = encode_rpn_bbox(anchor_bbox_output, object_bbox_output)
                        regression_output[y_output_pos, x_output_pos, :] = target_bbox

        for y_input_pos in range(input_shape[0]):
            for x_input_pos in range(input_shape[1]):
                iou = ious[y_input_pos, x_input_pos]
                if iou < max_false_iou:
                    # false sample
                    objectness_output[y_input_pos, x_input_pos, :] = [0, 1]
                elif iou > min_true_iou:
                    # true sample
                    objectness_output[y_input_pos, x_input_pos, :] = [1, 0]
                else:
                    # ignore sample
                    loss_mask[y_input_pos, x_input_pos] = False

    return regression_outputs, objectness_outputs, loss_masks


def rpn_output_nms(roi_boxes: typing.List[base.ROIBox]):
    pass


def decode_rpn_output(objectness_output: np.ndarray,
                      regression_output: np.ndarray,
                      anchor: typing.Tuple[float, float]) \
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
                center = (y_output_pos, x_output_pos)
                anchor_bbox = get_anchor_bbox(anchor, center)
                encoded_bbox = regression_output[y_output_pos, x_output_pos, :]
                decoded_bbox = decode_rpn_bbox(anchor_bbox, encoded_bbox)
                roi_box = base.ROIBox(objectness=objectness,
                                      bounding_box=decoded_bbox)
                roi_bboxes.append(roi_box)

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