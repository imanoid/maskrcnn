import tensorflow as tf
import data.squeezenet as squeezenet
import typing


class RCNNBuilder(squeezenet.SqueezeNetBuilder):
    def __init__(self, dtype=tf.float32):
        base.GraphBuilder.__init__(self, dtype)

    def add_roi_pooling(self,
                        input: tf.Tensor,
                        bboxes: tf.Tensor,
                        crop_size: tf.Tensor,
                        box_indices: tf.Tensor):
        crop = tf.image.crop_and_resize(image=input,
                                        boxes=bboxes,
                                        box_ind=box_indices,
                                        crop_size=crop_size)
        self.add_fire_module(self,
                             input,
                             256,
                             squeeze_ratio=0.125,  # percentage of squeeze kernels w.r.t. expand kernels
                             p_3x3=0.5,  # percentage of 3x3 kernels in expand kernels
                             skip_identity=False,  # if True, an identity skip connection is added
                             residual_input=None,
                             batch_norm=False,  # if True, batch normalization is added
                             is_training=None,  # pass variable indicating if network is training if using batch_norm
                             conv_keepprob=None)

    def add_simple_rpn_loss(self,
                            regression_predictions: typing.List[tf.Tensor],
                            regression_ground_truths: typing.List[tf.Tensor],
                            objectness_predictions: typing.List[tf.Tensor],
                            objectness_ground_truths: typing.List[tf.Tensor],
                            loss_masks: typing.List[tf.Tensor],
                            regression_loss_weight=tf.constant(10, tf.float32)):
        assert(len(regression_predictions) ==
               len(regression_ground_truths) ==
               len(objectness_predictions) ==
               len(objectness_ground_truths) ==
               len(loss_masks))
        n_anchors = len(regression_predictions)
        n_samples = regression_predictions[0].shape[0]

        regression_loss = tf.constant(0, tf.float32)
        objectness_loss = tf.constant(0, tf.float32)
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
            regression_loss += anchorwise_smooth_l1norm * regression_loss_weight / n_anchor_locations

            # objectness_loss
            objectness_prediction = objectness_predictions[i_anchor]
            objectness_ground_truth = objectness_ground_truths[i_anchor]

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=objectness_ground_truth,
                                                                    logits=objectness_prediction,
                                                                    dim=3) * loss_mask
            objectness_loss += cross_entropy / n_samples

        regression_loss = tf.reduce_sum(regression_loss)
        objectness_loss = tf.reduce_sum(objectness_loss)
        rpn_loss = objectness_loss + regression_loss

        return rpn_loss

    def add_simple_rpn_detector(self,
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