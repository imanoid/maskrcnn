import tensorflow as tf
import data.squeezenet as squeezenet
import typing
import data.coding as coding


class RCNNBuilder(squeezenet.SqueezeNetBuilder):
    def __init__(self, dtype=tf.float32):
        base.GraphBuilder.__init__(self, dtype)

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

    def rpn_bboxes(self,
                   objectness_nodes: tf.Tensor,
                   regression_nodes: tf.Tensor,
                   anchors: typing.List[typing.List[float, float]],
                   input_shape: typing.List[int, int],
                   iou_threshold: int=0.7,
                   max_bboxes: int=100) \
            -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """

        :param objectness_nodes:
        :param regression_nodes:
        :param anchors: list of lists containing (height, width)
        :param input_shape: list containing (height, width)
        :param iou_threshold:
        :param max_bboxes:
        :return: tuple containing (image indices, bounding boxes)
        """
        n_anchors = len(anchors)

        output_shape = objectness_nodes[0].shape[1:3]
        image_indices = tf.zeros([1], dtype=tf.int32)
        selected_boxes = tf.zeros([0, 4], dtype=self.dtype)
        for i_anchor in range(n_anchors):
            assert(objectness_nodes.shape[0] == regression_nodes.shape[0])

            coded_anchor_bboxes = coding.encode_anchor_bboxes(input_shape, output_shape, anchors[i_anchor])
            anchor_node = tf.constant(coded_anchor_bboxes.reshape(-1, 4), tf.float32)

            i_sample = tf.constant(0, tf.int32)

            def max_samples(i_sample: tf.Tensor, image_indices, selected_boxes):
                return tf.less(i_sample, tf.shape(objectness_nodes)[0])

            def sample_subgraph(i_sample: tf.Tensor, image_indices, selected_boxes):
                objectness_node = tf.reshape(objectness_nodes[i_anchor][i_sample, :, :, :], [-1, 2])
                objectness_scores = tf.nn.softmax(objectness_node, dim=1)[:, 0]

                regression_node = tf.reshape(regression_nodes[i_anchor][i_sample, :, :, :], [-1, 4])

                regression_corners = tf.add(tf.multiply(regression_node[:, 0:2], anchor_node[:, 2:4]),
                                            anchor_node[:, 0:2])
                regression_sizes = tf.multiply(tf.exp(tf.subtract(regression_node[:, 2:4], regression_node[:, 0:2])),
                                               anchor_node[:, 2:4])
                regression_boxes = tf.concat([regression_corners, tf.add(regression_corners, regression_sizes)], axis=1)
                selected_indices = tf.image.non_max_suppression(regression_boxes,
                                                                objectness_scores,
                                                                max_bboxes,
                                                                iou_threshold)

                selected_boxes = tf.concat([selected_boxes, regression_boxes[selected_indices, :]], 0)
                image_indices = tf.concat([image_indices, tf.ones_like(selected_indices, dtype=tf.int32)], 0)

                return [tf.add(i_sample, 1), image_indices, selected_boxes]

            _, image_indices, selected_boxes = tf.while_loop(max_samples,
                                                             sample_subgraph,
                                                             loop_vars=[i_sample, image_indices, selected_boxes],
                                                             shape_invariants=[i_sample.shape,
                                                                               tf.TensorShape([None]),
                                                                               tf.TensorShape([None, 4])])

        return image_indices, selected_boxes

    def roi_pooling(self,
                    input: tf.Tensor,
                    bboxes: tf.Tensor,
                    crop_size: tf.Tensor,
                    box_indices: tf.Tensor,
                    n_outputs: int=256):
        return tf.image.crop_and_resize(image=input,
                                        boxes=bboxes,
                                        box_ind=box_indices,
                                        crop_size=crop_size)

    def roi_classifier(self,
                       input: tf.Tensor,
                       n_outputs: int,
                       is_training: tf.Tensor,
                       conv_keepprob: tf.Tensor):
        n_hidden = input.shape[1]
        hidden = self.add_fire_module(input,
                                      n_hidden,
                                      skip_identity=True,
                                      batch_norm=True,
                                      is_training=is_training,
                                      conv_keepprob=conv_keepprob)
        output = self.add_fire_module(hidden,
                                      n_outputs,
                                      batch_norm=True,
                                      is_training=is_training,
                                      conv_keepprob=conv_keepprob
                                      )

    def simple_rpn_loss(self,
                        regression_predictions: typing.List[tf.Tensor],
                        regression_ground_truths: typing.List[tf.Tensor],
                        objectness_predictions: typing.List[tf.Tensor],
                        objectness_ground_truths: typing.List[tf.Tensor],
                        loss_masks: typing.List[tf.Tensor],
                        regression_loss_weight: int=10):
        assert(len(regression_predictions) ==
               len(regression_ground_truths) ==
               len(objectness_predictions) ==
               len(objectness_ground_truths) ==
               len(loss_masks))
        regression_loss_weight = tf.constant(regression_loss_weight, self.dtype)
        n_anchors = len(regression_predictions)
        n_samples = regression_predictions[0].shape[0]

        regression_loss = None
        objectness_loss = None
        for i_anchor in range(n_anchors):
            # regression loss
            regression_prediction = regression_predictions[i_anchor]
            regression_ground_truth = regression_ground_truths[i_anchor]
            loss_mask = loss_masks[i_anchor]
            n_anchor_locations = regression_prediction.shape[1] * regression_prediction.shape[2]

            diff = tf.subtract(regression_prediction, regression_ground_truth)
            abs_diff = tf.abs(diff)
            abs_diff_lt_1 = tf.less(abs_diff, 1)
            anchorwise_smooth_l1norm = tf.where(abs_diff_lt_1,
                                                0.5 * tf.square(abs_diff),
                                                abs_diff - 0.5) * loss_mask
            if regression_loss is None:
                regression_loss = anchorwise_smooth_l1norm * regression_loss_weight / n_anchor_locations
            else:
                regression_loss = tf.add(regression_loss, anchorwise_smooth_l1norm * regression_loss_weight / n_anchor_locations)

            # objectness_loss
            objectness_prediction = objectness_predictions[i_anchor]
            objectness_ground_truth = objectness_ground_truths[i_anchor]

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=objectness_ground_truth,
                                                                    logits=objectness_prediction,
                                                                    dim=3) * loss_mask
            if objectness_loss is None:
                objectness_loss = cross_entropy / n_samples
            else:
                objectness_loss = tf.add(objectness_loss, cross_entropy / n_samples)

        regression_loss = tf.reduce_sum(regression_loss)
        objectness_loss = tf.reduce_sum(objectness_loss)
        rpn_loss = objectness_loss + regression_loss

        return rpn_loss

    def simple_rpn_detector(self,
                            tail,
                            n_anchors,
                            n_channels=256,
                            batch_norm=False,
                            is_training=None,
                            conv_keepprob=None,
                            fc_keepprob=None
                            ):
        regression_nodes = list()
        objectness_nodes = list()

        if conv_keepprob is not None:
            tail = self.add_dropout_layer(tail, conv_keepprob)
        tail = self.add_conv_layer(tail,
                                   n_channels,
                                   kernel_size=1,
                                   batch_norm=batch_norm,
                                   is_training=is_training,
                                   bias=False)

        for i_anchor in range(n_anchors):
            with tf.name_scope("rpn_anchor{}".format(i_anchor)):
                regression_node = tail
                with tf.name_scope("regression".format(i_anchor)):
                    if fc_keepprob is not None:
                        regression_node = self.add_dropout_layer(regression_node, fc_keepprob)
                    regression_node = self.add_conv_layer(regression_node,
                                                          4,
                                                          kernel_size=7,
                                                          batch_norm=batch_norm,
                                                          is_training=is_training,
                                                          bias=False)
                    regression_nodes.append(regression_node)

                objectness_node = tail
                with tf.name_scope("objectness".format(i_anchor)):
                    if fc_keepprob is not None:
                        objectness_node = self.add_dropout_layer(objectness_node, fc_keepprob)
                    objectness_node = self.add_conv_layer(objectness_node,
                                                          2,
                                                          kernel_size=7,
                                                          batch_norm=batch_norm,
                                                          is_training=is_training,
                                                          bias=False)
                    objectness_nodes.append(objectness_node)

        return objectness_nodes, regression_nodes