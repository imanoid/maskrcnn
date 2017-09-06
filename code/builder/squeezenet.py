import tensorflow as tf
import data.base as base


class SqueezeNetBuilder(base.GraphBuilder):
    def __init__(self, dtype=tf.float32):
        base.GraphBuilder.__init__(self, dtype)

    def add_trunk(self,
                  root_output_node,  # input graph node
                  pooled_firemodules=None,  # modules at which to pool
                  first_conv_ksize=3,  # kernel size of first convolution
                  first_conv_stride=2,  # stride of first convolution
                  first_conv_kernels=96,  # number of kernels in first convolution layer
                  n_modules=10,  # number of modules
                  base_outputs=128,  # n_outputs of first fire module
                  incr_outputs=128,  # step by which to increase output depth every freq units
                  freq=2,  # frequency at which num of outputs is increased by incr_outputs
                  squeeze_ratio=0.25,  # percentage of squeeze kernels in all kernels (squeeze and expand)
                  p_3x3=0.5,  # percentage of 3x3 kernels in expand kernels
                  skip_identity=True,  # true if residual identity connections should be added
                  batch_norm=False,  # if True, batch normalization is added
                  is_training=None,  # pass variable indicating if network is training if using batch_norm
                  input_keepprob=None,  # input keep probability for dropout
                  conv_keepprob=None  # conv keep probability for dropout
                  ):
        if pooled_firemodules is None:
            pooled_firemodules = [1, 3, 5]
        node = root_output_node

        with tf.name_scope("Conv1"):
            if input_keepprob is not None:
                node = self.add_dropout_layer(node, input_keepprob)
            node = self.add_conv_layer(node,
                                       n_outputs=first_conv_kernels,
                                       kernel_size=first_conv_ksize,
                                       strides=first_conv_stride,
                                       batch_norm=batch_norm,
                                       is_training=is_training,
                                       bias=False)
        if 1 in pooled_firemodules:
            node = self.add_maxpooling_layer(node)

        segment_tails = []
        n_module_outputs = base_outputs

        for i_module in range(2, n_modules):
            if i_module % freq == 0 and i_module > 0:
                n_module_outputs += incr_outputs
            with tf.name_scope("FireModule{}".format(i_module)):
                node = self.add_fire_module(node,
                                            n_module_outputs,
                                            squeeze_ratio=squeeze_ratio,
                                            p_3x3=p_3x3,
                                            skip_identity=skip_identity,
                                            residual_input=None,
                                            batch_norm=batch_norm,
                                            is_training=is_training,
                                            conv_keepprob=conv_keepprob,)

                if i_module in pooled_firemodules:
                    node = self.add_maxpooling_layer(node, kernel_size=3, strides=2)
                    segment_tails.append(node)
                elif i_module == n_modules - 1:
                    segment_tails.append(node)

        return segment_tails

    def add_classifier_head(self,
                            trunk_output_node,  # output from the trunk
                            n_outputs,  # number of outputs
                            fc_keepprob=None,  # fc keep probability for dropout
                            batch_norm=False,
                            is_training=None
                            ):
        node = trunk_output_node
        with tf.name_scope("Conv10"):
            if fc_keepprob is not None:
                node = self.add_dropout_layer(node, fc_keepprob)
            node = self.add_conv_layer(node,
                                       n_outputs,
                                       kernel_size=1,
                                       batch_norm=batch_norm,
                                       is_training=is_training)
        return self.add_fc_avgpooling_layer(node)

    def add_upsampling_pyramid(self,
                               segment_tails,  # tails of the trunk segments outputs
                               squeeze_ratio=.125,
                               p_3x3=0.5,
                               conv_keepprob=None,
                               batch_norm=False,
                               is_training=None,
                               skip_identity=True
                               ):
        node = None

        feature_maps = []

        for i_segment, segment_tail in enumerate(reversed(segment_tails)):
            segment_shape = segment_tail.shape.as_list()
            n_outputs = segment_shape[3]

            if len(feature_maps) == 0:
                feature_maps.append(segment_tail)
                node = self.add_upsampling_layer(segment_tail)
            else:
                with tf.name_scope("UpsamplingSegment"):
                    node = self.add_fire_module(node,
                                                n_outputs=n_outputs,
                                                residual_input=segment_tail,
                                                squeeze_ratio=squeeze_ratio,
                                                p_3x3=p_3x3,
                                                batch_norm=batch_norm,
                                                is_training=is_training,
                                                skip_identity=skip_identity,
                                                conv_keepprob=conv_keepprob)
                    node = self.add_upsampling_layer(node)
            feature_maps.append(node)
        return feature_maps

    def add_fire_module(self,
                        input_node,  # input graph node
                        n_outputs,  # total number of outputs
                        squeeze_ratio=0.125,  # percentage of squeeze kernels w.r.t. expand kernels
                        p_3x3=0.5,  # percentage of 3x3 kernels in expand kernels
                        skip_identity=False,  # if True, an identity skip connection is added
                        residual_input=None,
                        batch_norm=False,  # if True, batch normalization is added
                        is_training=None,  # pass variable indicating if network is training if using batch_norm
                        conv_keepprob=None  # conv keep probability for dropout
                        ):
        assert (type(n_outputs) == int)

        with tf.name_scope("FireModule"):
            input_shape = input_node.shape.as_list()

            input_channels = input_shape[3]

            squeeze_input_node = input_node

            if batch_norm:
                squeeze_input_node = self.add_batch_norm(input_node, is_training=is_training, global_norm=True)
            squeeze_input_node = tf.nn.relu(squeeze_input_node)

            assert (float(n_outputs) * squeeze_ratio * p_3x3 == int(float(n_outputs) * squeeze_ratio * p_3x3))

            s_1x1 = int(float(n_outputs) * squeeze_ratio)  # squeeze depth
            e_3x3 = int(float(n_outputs) * p_3x3)  # number of expand 3x3 kernels
            e_1x1 = int(n_outputs - e_3x3)  # number of expand 1x1 kernels

            if conv_keepprob is not None:
                squeeze_input_node = self.add_dropout_layer(squeeze_input_node, conv_keepprob)

            node = self.add_conv_layer(squeeze_input_node, s_1x1, kernel_size=1, batch_norm=batch_norm,
                                       is_training=is_training, bias=False)

            if conv_keepprob is not None:
                node = self.add_dropout_layer(node, conv_keepprob)

            branch_1x1 = self.add_conv_layer(node, e_1x1, kernel_size=1, activation=None)
            branch_3x3 = self.add_conv_layer(node, e_3x3, kernel_size=3, activation=None)

            output_node = tf.concat([branch_1x1, branch_3x3], 3)

            if residual_input is not None:
                output_node = tf.add(residual_input, output_node)
            if input_channels == n_outputs and skip_identity:
                output_node = tf.add(input_node, output_node)

            return output_node
