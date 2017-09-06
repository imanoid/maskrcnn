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

    def _add_conv_layer(self,
                        node,
                        function,
                        n_outputs,
                        kernel_size,
                        strides,
                        activation,
                        batch_norm,
                        split,
                        is_training,
                        bias):
        with tf.name_scope("Convolution_%dx%d" % (kernel_size, kernel_size)):
            if split:
                stddev = 2.0 / np.sqrt(kernel_size * int(node.shape[3]))
                kernel1 = tf.Variable(
                    tf.truncated_normal([1, kernel_size, int(node.shape[3]), n_outputs], stddev=stddev,
                                        dtype=self.dtype))
                node = function(node, kernel1, strides=[1, strides, strides, 1], padding="SAME")
                if activation != None:
                    node = activation(node)

                stddev = 2.0 / np.sqrt(kernel_size * n_outputs)
                kernel2 = tf.Variable(
                    tf.truncated_normal([kernel_size, 1, n_outputs, n_outputs], stddev=stddev, dtype=self.dtype))
                node = function(node, kernel2, strides=[1, strides, strides, 1], padding="SAME")
            else:
                stddev = 2.0 / np.sqrt(kernel_size * kernel_size * int(node.shape[3]))
                kernel = tf.Variable(
                    tf.truncated_normal([kernel_size, kernel_size, int(node.shape[3]), n_outputs], stddev=stddev,
                                        dtype=self.dtype))
                node = function(node, kernel, strides=[1, strides, strides, 1], padding="SAME")
            if batch_norm:
                node = self.add_batch_norm(node, is_training, True)
            elif bias:
                bias = tf.Variable(tf.random_normal([n_outputs], dtype=self.dtype))
                node = tf.nn.bias_add(node, bias)

            if activation != None:
                return activation(node)
            else:
                return node

    def add_conv_layer(self,
                       node,
                       n_outputs,
                       kernel_size=3,
                       strides=1,
                       activation=tf.nn.relu,
                       batch_norm=False,
                       split=False,
                       is_training=None,
                       bias=True):
        return self._add_conv_layer(node, tf.nn.conv2d, n_outputs, kernel_size, strides, activation, batch_norm, split,
                                    is_training, bias)

    def add_deconv_layer(self,
                         node,
                         n_outputs,
                         kernel_size=3,
                         strides=1,
                         activation=tf.nn.relu,
                         batch_norm=False,
                         split=False,
                         is_training=None,
                         bias=True):
        return self._add_conv_layer(node, tf.nn.conv2d_transpose, n_outputs, kernel_size, strides, activation,
                                    batch_norm, split, is_training, bias)

    def add_fc_layer(self,
                     node,
                     n_outputs,
                     activation=None,
                     batch_norm=False,
                     fully_conv=True,
                     is_training=None,
                     bias=True):
        with tf.name_scope("FullyConnected"):
            if fully_conv:
                # conv
                stddev = 2.0 / np.sqrt(int(node.shape[1]) * int(node.shape[2]) * int(node.shape[3]))
                kernel = tf.Variable(
                    tf.truncated_normal([int(node.shape[1]), int(node.shape[2]), int(node.shape[3]), n_outputs],
                                        stddev=stddev, dtype=self.dtype))
                node = tf.nn.conv2d(node, kernel, strides=[1, 1, 1, 1], padding="VALID")
            else:
                if len(node.shape) == 4:
                    node = tf.reshape(node, [-1, int(node.shape[1]) * int(node.shape[2]) * int(node.shape[3])])
                stddev = 2.0 / np.sqrt(node.shape[1])
                weights = tf.Variable(
                    tf.truncated_normal([int(node.shape[1]), n_outputs], stddev=stddev, dtype=self.dtype))
                node = tf.matmul(node, weights)

            if batch_norm:
                node = self.add_batch_norm(node, is_training, True)
            elif bias:
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