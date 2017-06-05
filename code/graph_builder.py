import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

# 'backport' from current tensorflow version to support gradient descent with MaxPoolWithArgmax
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                 grad,
                                                 op.outputs[1],
                                                 op.get_attr("ksize"),
                                                 op.get_attr("strides"),
                                                 padding=op.get_attr("padding"))

class GraphBuilder(object):
    def __init__(self, dtype=tf.float32):
        self.dtype = dtype

    def _add_conv_layer(self, node, function, n_outputs, kernel_size, strides, activation, batch_norm, split, is_training):
        with tf.name_scope("Convolution_%dx%d" % (kernel_size, kernel_size)):
            if split:
                stddev = 2.0 / np.sqrt(kernel_size * int(node.shape[3]))
                kernel1 = tf.Variable(tf.truncated_normal([1, kernel_size, int(node.shape[3]), n_outputs], stddev=stddev, dtype=self.dtype))
                node = function(node, kernel1, strides=[1, strides, strides, 1], padding="SAME")
                if activation != None:
                    node = activation(node)

                stddev = 2.0 / np.sqrt(kernel_size * n_outputs)
                kernel2 = tf.Variable(tf.truncated_normal([kernel_size, 1, n_outputs, n_outputs], stddev=stddev, dtype=self.dtype))
                node = function(node, kernel2, strides=[1, strides, strides, 1], padding="SAME")
            else:
                stddev = 2.0 / np.sqrt(kernel_size * kernel_size * int(node.shape[3]))
                kernel = tf.Variable(
                    tf.truncated_normal([kernel_size, kernel_size, int(node.shape[3]), n_outputs], stddev=stddev, dtype=self.dtype))
                node = function(node, kernel, strides=[1, strides, strides, 1], padding="SAME")
            if batch_norm:
                node = self.add_batch_norm(node, is_training, True)
            else:
                bias = tf.Variable(tf.random_normal([n_outputs], dtype=self.dtype))
                node = tf.nn.bias_add(node, bias)

            if activation != None:
                return activation(node)
            else:
                return node

    def add_conv_layer(self, node, n_outputs, kernel_size=3, strides=1, activation=tf.nn.relu, batch_norm=False, split=False, is_training=None):
        return self._add_conv_layer(node, tf.nn.conv2d, n_outputs, kernel_size, strides, activation, batch_norm, split, is_training)

    def add_deconv_layer(self, node, n_outputs, kernel_size=3, strides=1, activation=tf.nn.relu, batch_norm=False, split=False, is_training=None):
        return self._add_conv_layer(node, tf.nn.conv2d_transpose, n_outputs, kernel_size, strides, activation, batch_norm, split, is_training)

    def add_fc_layer(self, node, n_outputs, activation=None, batch_norm=False, fully_conv=True, is_training=None):
        with tf.name_scope("FullyConnected"):
            if fully_conv:
                # conv
                stddev = 2.0 / np.sqrt(int(node.shape[1]) * int(node.shape[2]) * int(node.shape[3]))
                kernel = tf.Variable(tf.truncated_normal([int(node.shape[1]), int(node.shape[2]), int(node.shape[3]), n_outputs], stddev=stddev, dtype=self.dtype))
                node = tf.nn.conv2d(node, kernel, strides=[1, 1, 1, 1], padding="VALID")
            else:
                if len(node.shape) == 4:
                    node = tf.reshape(node, [-1, int(node.shape[1]) * int(node.shape[2]) * int(node.shape[3])])
                stddev = 2.0 / np.sqrt(node.shape[1])
                weights = tf.Variable(tf.truncated_normal([int(node.shape[1]), n_outputs], stddev=stddev, dtype=self.dtype))
                node = tf.matmul(node, weights)

            if batch_norm:
                node = self.add_batch_norm(node, is_training, True)
            else:
                bias = tf.Variable(tf.random_normal([n_outputs], dtype=self.dtype))
                node = tf.nn.bias_add(node, bias)

            if activation == None:
                return node
            else:
                return activation(node)

    def _add_pooling_layer(self, node, function, kernel_size=2, strides=2, padding="SAME"):
        return function(node, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1], padding=padding)

    def add_maxpooling_layer(self, node, kernel_size=2, strides=2):
        with tf.name_scope("MaxPooling_%dx%d" % (kernel_size, kernel_size)):
            return self._add_pooling_layer(node, tf.nn.max_pool, kernel_size=kernel_size, strides=strides)

    def add_avgpooling_layer(self, node, kernel_size=2, strides=2):
        with tf.name_scope("AvgPooling_%dx%d" % (kernel_size, kernel_size)):
            return self._add_pooling_layer(node, tf.nn.avg_pool, kernel_size=kernel_size, strides=strides)

    def add_fc_maxpooling_layer(self, node):
        with tf.name_scope("MaxPooling_FC"):
            kernel_size = int(node.shape[1])
            return self._add_pooling_layer(node, tf.nn.max_pool, kernel_size=kernel_size, padding="VALID")

    def add_fc_avgpooling_layer(self, node):
        with tf.name_scope("AvgPooling_FC"):
            kernel_size = int(node.shape[1])
            return self._add_pooling_layer(node, tf.nn.avg_pool, kernel_size=kernel_size, padding="VALID")

    def add_unpooling_layer(self, node, argmax=None, kernel_size=2):
        with tf.name_scope("Unpooling_%dx%d" % (kernel_size, kernel_size)):
            if argmax == None:
                pass
            else:
                kernel_elements = kernel_size ** 2
                input_shape = node.shape.as_list()
                input_elements = np.prod(input_shape[1:])

                argmax_shape = argmax.shape.as_list()
                argmax_elements = np.prod(argmax_shape[1:])

                # tf.sparse_to_dence requires the indices of the values (i.e. argmax) to be sorted in ascending order, therefore we do this...
                argmax, argmax_permutation = tf.nn.top_k(tf.reshape(argmax, [-1, argmax_elements]), k=argmax_elements)
                argmax = tf.reverse(argmax, axis=[1])
                argmax_permutation = tf.reverse(argmax_permutation, axis=[1])

                values = tf.reshape(node, [-1, input_elements])

                shape = [input_elements * kernel_elements]

                batch_sparse_to_dense = lambda x: tf.sparse_to_dense(x[0], shape, tf.gather(x[1], x[2]))

                # create flat outputs
                node = tf.map_fn(batch_sparse_to_dense, (argmax, values, argmax_permutation), dtype=tf.float32)

                # reshape to sparse output matrix
                node = tf.reshape(node,
                                         [-1, input_shape[1] * kernel_size, input_shape[2] * kernel_size, input_shape[3]])

                return node

    def add_upsampling_layer(self, node, factor=2):
        node_shape = node.shape.as_list()
        with tf.name_scope("Upsampling"):
            return tf.image.resize_nearest_neighbor(node, [node_shape[1] * factor, node_shape[2] * factor])

    def add_dropout_layer(self, node, keep_prob):
        with tf.name_scope("Dropout"):
            return tf.nn.dropout(node, keep_prob)

    def add_batch_norm(self, node, is_training, global_norm=True):
        assert not tf.get_variable_scope().reuse, "Cannot create stream-unique averages if reuse flag is ON"

        with tf.name_scope("BatchNorm"):
            node_shape = node.shape
            if global_norm:
                moments_shape = [1, 1, 1, node.shape[3]]
                moments_axes = [0, 1, 2]
            else:
                moments_shape = [1, node.shape[1], node.shape[2], node.shape[3]]
                moments_axes = [0]

            moving_mean = tf.Variable(tf.zeros(moments_shape, dtype=self.dtype))
            moving_var = tf.Variable(tf.ones(moments_shape, dtype=self.dtype))

            offset = tf.Variable(tf.zeros(moments_shape, dtype=self.dtype))
            scale = tf.ones_like(tf.ones(moments_shape, dtype=self.dtype))


            def training():
                [batch_mean, batch_var] = tf.nn.moments(node, moments_axes)

                update_mm = moving_averages.assign_moving_average(moving_mean, batch_mean, .99)
                update_mv = moving_averages.assign_moving_average(moving_var, batch_var, .99)

                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mm)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mv)

                return tf.nn.batch_normalization(node, batch_mean, batch_var, offset, scale, variance_epsilon=1e-8)

            def testing():
                return tf.nn.batch_normalization(node, moving_mean, moving_var, offset, scale, variance_epsilon=1e-8)

            node = tf.cond(is_training, training, testing)
            node.set_shape(node_shape)
            return node

class SqueezeNetBuilder(GraphBuilder):
    def __init__(self, dtype = tf.float32):
        GraphBuilder.__init__(self, dtype)

    def add_root(self,
                 input_node,  # input graph node
                   first_conv_ksize = 7,  # kernel size of first convolution
                   first_conv_stride = 2,  # stride of first convolution
                   first_conv_kernels = 96,  # number of kernels in first convolution layer
                   input_keepprob = None,  # input keep probability for dropout
                   batch_norm = False,
                 is_training = None
                 ):
        with tf.name_scope("InputSegment"):
            node = input_node
            if input_keepprob != None:
                node = self.add_dropout_layer(node, input_keepprob)
            node = self.add_conv_layer(node, first_conv_kernels, kernel_size=first_conv_ksize, strides=first_conv_stride, batch_norm=batch_norm, is_training=is_training)
            node = self.add_maxpooling_layer(node)
            return node

    def add_trunk(self,
                  root_output_node,  # input graph node
                  n_modules,  # number of modules
                  pooled_firemodules,  # modules at which to pool
                  base_outputs = 128,  # n_outputs of first fire module
                  incr_outputs = 128,  # step by which to increase output depth every freq units
                  freq = 2,  # frequency at which num of outputs is increased by incr_outputs
                  squeeze_ratio = 0.25,  # percentage of squeeze kernels in all kernels (squeeze and expand)
                  p_3x3 = 0.5,  # percentage of 3x3 kernels in expand kernels
                  skip_identity = True,  # true if residual identity connections should be added
                  batch_norm = False,  # if True, batch normalization is added
                  is_training = None,  # pass variable indicating if network is training if using batch_norm
                  conv_keepprob = None  # conv keep probability for dropout
                  ):

        segment_tails = []
        node = root_output_node
        n_module_outputs = base_outputs

        pool_outputs = False
        pooled_last_outputs = False

        for i_module in range(2, n_modules):
            if i_module % freq == 0 and i_module > 0:
                n_module_outputs += incr_outputs
            pooled_last_outputs = pool_outputs
            pool_outputs = (i_module in pooled_firemodules)
            with tf.name_scope("TrunkSegment"):
                node = self.add_fire_module(node,
                                            n_module_outputs,
                                            squeeze_ratio = squeeze_ratio,
                                            p_3x3 = p_3x3,
                                            skip_identity = skip_identity,
                                            residual_input = None,
                                            batch_norm = batch_norm,
                                            is_training = is_training,
                                            conv_keepprob = conv_keepprob,
                                            activate_input = not pooled_last_outputs,
                                            activate_output = pool_outputs)

                if pool_outputs:
                    node = self.add_maxpooling_layer(node)
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

        with tf.name_scope("ClassifierSegment"):
            if fc_keepprob != None:
                node = self.add_dropout_layer(trunk_output_node, fc_keepprob)
            node = self.add_conv_layer(node, n_outputs, kernel_size=1, batch_norm=batch_norm, is_training=is_training)
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
                    node = self.add_fire_module(node, n_outputs, residual_input=segment_tail, squeeze_ratio=squeeze_ratio, p_3x3=p_3x3, batch_norm=batch_norm, is_training=is_training, skip_identity=skip_identity, activate_input=True, activate_output=True, conv_keepprob=conv_keepprob)
                    node = self.add_upsampling_layer(node)

        return node

    def add_fire_module(self,
                        input_node,  # input graph node
                        n_outputs,  # total number of outputs
                        squeeze_ratio = 0.125,  # percentage of squeeze kernels w.r.t. expand kernels
                        p_3x3 = 0.5,  # percentage of 3x3 kernels in expand kernels
                        skip_identity = False,  # if True, an identity skip connection is added
                        residual_input = None,
                        batch_norm = False,  # if True, batch normalization is added
                        is_training = None,  # pass variable indicating if network is training if using batch_norm
                        conv_keepprob = None,  # conv keep probability for dropout
                        activate_input = True,  # true of the module comes after a pooling layer
                        activate_output = False  # true if the module comes before a pooling layer
                        ):
        assert(type(n_outputs) == int)

        with tf.name_scope("FireModule"):
            input_shape = input_node.shape.as_list()

            input_channels = input_shape[3]

            squeeze_input_node = input_node

            if activate_input:
                if batch_norm:
                    squeeze_input_node = self.add_batch_norm(input_node, is_training=is_training, global_norm=True)
                squeeze_input_node = tf.nn.relu(squeeze_input_node)

            assert(float(n_outputs) * squeeze_ratio * p_3x3 == int(float(n_outputs) * squeeze_ratio * p_3x3))

            s_1x1 = int(float(n_outputs) * squeeze_ratio) # squeeze depth
            e_3x3 = int(float(n_outputs) * p_3x3) # number of expand 3x3 kernels
            e_1x1 = int(n_outputs - e_3x3) # number of expand 1x1 kernels

            if conv_keepprob is not None:
                squeeze_input_node = self.add_dropout_layer(squeeze_input_node, conv_keepprob)

            node = self.add_conv_layer(squeeze_input_node, s_1x1, kernel_size=1, batch_norm=batch_norm, is_training=is_training)

            if conv_keepprob is not None:
                node = self.add_dropout_layer(node, conv_keepprob)

            branch_1x1 = self.add_conv_layer(node, e_1x1, kernel_size=1, activation=None)
            branch_3x3 = self.add_conv_layer(node, e_3x3, kernel_size=3, activation=None)

            output_node = tf.concat([branch_1x1, branch_3x3], 3)

            if residual_input is not None:
                output_node = tf.add(residual_input, output_node)
            if input_channels == n_outputs and skip_identity:
                output_node = tf.add(input_node, output_node)

            if activate_output:
                if batch_norm:
                    output_node = self.add_batch_norm(output_node, is_training=is_training, global_norm=True)
                output_node = tf.nn.relu(output_node)

            return output_node