import tensorflow as tf
import numpy as np
import builder.base as base


class ShuffleNetBuilder(base.GraphBuilder):
    def __init__(self, dtype=tf.float32):
        base.GraphBuilder.__init__(self, dtype)
    
    def add_trunk(self,
                  node,  # input
                  first_conv_ksize=3,  # kernel size of first convolution
                  first_conv_stride=2,  # stride of first convolution
                  first_conv_kernels=24,  # number of kernels in first convolution layer
                  first_pool_ksize=3,  # kernel size of first maxpool
                  first_pool_stride=2,  # stride of first maxpool
                  batch_norm=False,  # if True, batch normalization is added
                  is_training=None,  # pass variable indicating if network is training
                  shuffle_segments=[4, 8, 4],
                  n_groups=8,
                  base_channels=384,
                  bottleneck_ratio=.25,
                  input_keepprob=None,  # input keep probability for dropout
                  conv_keepprob=None  # conv keep probability for dropout
                  ):
        assert(base_channels // n_groups == base_channels / n_groups)
        assert((base_channels / n_groups) * .25 == int((base_channels / n_groups) * .25))
        
        with tf.name_scope("Segment1"):
            if input_keepprob is not None:
                node = self.add_dropout_layer(node, input_keepprob)
            node = self.add_conv_layer(node,
                                       n_outputs=first_conv_kernels,
                                       kernel_size=first_conv_ksize,
                                       strides=first_conv_stride,
                                       batch_norm=batch_norm,
                                       is_training=is_training,
                                       activation=None,
                                       bias=False)
            node = self.add_maxpooling_layer(node,
                                             kernel_size=first_pool_ksize,
                                             strides=first_pool_stride)
        
        n_channels = base_channels
        
        segment_tails = list()
        
        for i_segment, n_modules in enumerate(shuffle_segments):
            with tf.name_scope("ShuffleNetSegment{}".format(i_segment)):
                for i_module in range(n_modules):
                    if i_module == 0:
                        stride = 2
                    else:
                        stride = 1

                    if i_segment == i_module == 0:
                        nodes = self.add_first_shuffle_module(node=node,
                                                              n_groups=n_groups,
                                                              is_training=is_training,
                                                              bottleneck_ratio=bottleneck_ratio,
                                                              stride=stride,
                                                              n_channels=n_channels,
                                                              conv_keepprob=conv_keepprob)
                    else:
                        with tf.name_scope("ShuffleNetModule{}".format(i_module)):
                            nodes = self.add_shuffle_module(nodes,
                                                            batch_norm=batch_norm,
                                                            is_training=is_training,
                                                            bottleneck_ratio=bottleneck_ratio,
                                                            stride=stride,
                                                            n_channels=n_channels,
                                                            conv_keepprob=conv_keepprob)
                if len(nodes) == 1:
                    segment_tail = nodes[0]
                else:
                    segment_tail = tf.concat(nodes, axis=3)
                
                if batch_norm:
                    segment_tail = self.add_batch_norm(segment_tail, is_training=is_training, global_norm=True)
                segment_tails.append(tf.nn.relu(segment_tail))
            n_channels *= 2
        return segment_tails
    
    def add_upsampling_pyramid(self,
                               segment_tails,  # tails of the trunk segments outputs
                               shuffle_segments=None,
                               bottleneck_ratio=.25,
                               n_groups=8,
                               conv_keepprob=None,
                               batch_norm=False,
                               is_training=None
                               ):
        pass
    
    def add_shuffle_module(self,
                           nodes,  # list of input nodes
                           n_channels,  # number of output channels
                           batch_norm=False,  # if True, batch normalization is added
                           is_training=None,  # pass variable indicating if network is training
                           bottleneck_ratio=0.25,  # channel reduction ratio
                           stride=1,  # stride of 3x3 conv
                           conv_keepprob=None  # conv keep probability for dropout
                           ):
        n_groups = len(nodes)
        
        if stride == 1:            
            n_branch_output_channels = n_channels
        elif stride == 2:
            n_input_channels = nodes[0].shape[3].value * n_groups
            n_branch_output_channels = n_channels - n_input_channels
        else:
            raise ValueError("Stride {} is invalid for shuffle module!")
            
        n_hidden_channels = n_branch_output_channels * bottleneck_ratio
        
        assert(n_hidden_channels == int(n_hidden_channels))
        
        input_nodes = nodes
        
        with tf.name_scope("InputActivation"):
            if batch_norm:
                nodes = self.add_group_batchnorm(nodes, is_training=is_training, global_norm=True)
            
            nodes = self.add_group_relu(nodes)
        
        with tf.name_scope("ReductionGroupConv"):
            if conv_keepprob is not None:
                node = self.add_group_dropout_layer(nodes, conv_keepprob)
            
            nodes = self.add_group_conv_layer(nodes,
                                              n_outputs=n_hidden_channels,
                                              kernel_size=1,
                                              strides=1,
                                              batch_norm=batch_norm,
                                              is_training=is_training,
                                              bias=False)
                                              
        with tf.name_scope("InnerConvShuffle"):
            if len(nodes) == 1:
                node = nodes[0]
            else:
                node = tf.concat(nodes, axis=3, name="ChannelMerge")
        
            if conv_keepprob is not None:
                node = self.add_dropout_layer(node, conv_keepprob)
            
            node = self.add_depthwise_conv_layer(node,
                                                 kernel_size=3,
                                                 strides=stride,
                                                 activation=None,
                                                 batch_norm=batch_norm,
                                                 is_training=is_training,
                                                 bias=False)
            node = self.add_channel_shuffle(node, n_groups)
            nodes = self.add_channel_split(node, n_groups)
        
        with tf.name_scope("ExpansionGroupConv"):
            if conv_keepprob is not None:
                node = self.add_group_dropout_layer(nodes, conv_keepprob)
            
            nodes = self.add_group_conv_layer(nodes,
                                              n_outputs=n_branch_output_channels,
                                              kernel_size=1,
                                              strides=1,
                                              activation=None,
                                              bias=False)
        
        with tf.name_scope("ResidualConnection"):
            if stride == 1:
                nodes = self.add_group_add_layer(input_nodes, nodes)
            else:
                input_nodes = self.add_group_avgpooling_layer(input_nodes,
                                                              kernel_size=3, 
                                                              strides=2)
                nodes = self.add_group_concat_layer(input_nodes, nodes)
        
        return nodes

    def add_first_shuffle_module(self,
                                 node,  # input node
                                 n_groups,  # number of output groups
                                 n_channels,  # number of output channels
                                 batch_norm=False,  # if True, batch normalization is added
                                 is_training=None,  # pass variable indicating if network is training
                                 bottleneck_ratio=0.25,  # channel reduction ratio
                                 stride=1,  # stride of 3x3 conv
                                 conv_keepprob=None  # conv keep probability for dropout
                                 ):
        if stride == 1:
            n_branch_output_channels = n_channels
        elif stride == 2:
            n_input_channels = node.shape[3].value
            n_branch_output_channels = n_channels - n_input_channels
        else:
            raise ValueError("Stride {} is invalid for shuffle module!")

        n_hidden_channels = n_branch_output_channels * bottleneck_ratio

        assert (n_hidden_channels == int(n_hidden_channels))

        n_hidden_channels = int(n_hidden_channels)

        input_nodes = self.add_channel_split(node, n_groups)

        with tf.name_scope("InputActivation"):
            if batch_norm:
                node = self.add_batch_norm(node, is_training=is_training, global_norm=True)

            node = tf.nn.relu(node)

        with tf.name_scope("ReductionConv"):
            if conv_keepprob is not None:
                node = self.add_dropout_layer(node, conv_keepprob)
            node = self.add_conv_layer(node,
                                       n_outputs=n_hidden_channels,
                                       kernel_size=1,
                                       strides=1,
                                       batch_norm=batch_norm,
                                       is_training=is_training,
                                       bias=False)

        with tf.name_scope("InnerConvShuffle"):
            if conv_keepprob is not None:
                node = self.add_dropout_layer(node, conv_keepprob)

            node = self.add_depthwise_conv_layer(node,
                                                 kernel_size=3,
                                                 strides=stride,
                                                 activation=None,
                                                 batch_norm=batch_norm,
                                                 is_training=is_training,
                                                 bias=False)
            nodes = self.add_channel_split(node, n_groups)

        with tf.name_scope("ExpansionGroupConv"):
            if conv_keepprob is not None:
                node = self.add_group_dropout_layer(nodes, conv_keepprob)

            nodes = self.add_group_conv_layer(nodes,
                                              n_outputs=n_branch_output_channels,
                                              kernel_size=1,
                                              strides=1,
                                              activation=None,
                                              bias=False)

        with tf.name_scope("ResidualConnection"):
            if stride == 1:
                nodes = self.add_group_add_layer(input_nodes, nodes)
            else:
                input_nodes = self.add_group_avgpooling_layer(input_nodes,
                                                              kernel_size=3,
                                                              strides=2)
                nodes = self.add_group_concat_layer(input_nodes, nodes)

        return nodes

    def add_classifier_head(self,
                            trunk_output_node,  # output from the trunk
                            n_outputs,  # number of outputs
                            fc_keepprob=None,  # fc keep probability for dropout
                            ):
        with tf.name_scope("ClassifierHead"):
            node = self.add_fc_avgpooling_layer(trunk_output_node)
            if fc_keepprob is not None:
                node = self.add_dropout_layer(node, fc_keepprob)
            node = self.add_fc_layer(node, n_outputs=n_outputs, bias=False, fully_conv=False)
        return node
        
    #  emulate groupwise convolutions and other operations
    def add_group_conv_layer(self,
                             nodes,
                             n_outputs,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             batch_norm=False,
                             split=False,
                             is_training=None,
                             bias=True):
        n_groups = len(nodes)
        n_created_outputs = 0

        with tf.name_scope("GroupedConvolution_%dx%d" % (kernel_size, kernel_size)):
            output_nodes = list()
            for node in nodes:
                n_group_outputs = min(int(np.ceil(n_outputs / n_groups)), n_outputs - n_created_outputs)
                output_nodes.append(self.add_conv_layer(node,
                                                        n_group_outputs,
                                                        kernel_size=kernel_size,
                                                        strides=strides,
                                                        activation=activation,
                                                        batch_norm=batch_norm,
                                                        split=split,
                                                        is_training=is_training,
                                                        bias=bias))
                n_created_outputs += n_group_outputs
        return output_nodes
    
    def add_group_add_layer(self,
                            nodes1,
                            nodes2):
        assert(len(nodes1) == len(nodes2))
        with tf.name_scope("GroupAdd"):
            output_nodes = list()
            for i_node in range(len(nodes1)):
                output_nodes.append(tf.add(nodes1[i_node], nodes2[i_node]))
        
        return output_nodes
    
    def add_group_concat_layer(self,
                               nodes1,
                               nodes2):
        assert(len(nodes1) == len(nodes2))
        with tf.name_scope("GroupConcat"):
            output_nodes = list()
            for i_node in range(len(nodes1)):
                output_nodes.append(tf.concat([nodes1[i_node], nodes2[i_node]], axis=3))
        
        return output_nodes
    
    def add_group_relu(self,
                       nodes):
        with tf.name_scope("GroupRelu"):
            output_nodes = list()
            for node in nodes:
                output_nodes.append(tf.nn.relu(node))
        return output_nodes
    
    def add_group_avgpooling_layer(self, 
                                   nodes, 
                                   kernel_size=2, 
                                   strides=2):
        with tf.name_scope("GroupAvgPool"):
            output_nodes = list()
            for node in nodes:
                output_nodes.append(self.add_avgpooling_layer(node, kernel_size=kernel_size, strides=strides))
        return output_nodes
    
    def add_group_batchnorm(self,
                            nodes,
                            is_training=None,
                            global_norm=True):
        with tf.name_scope("GroupBatchnorm"):
            output_nodes = list()
            for node in nodes:
                output_nodes.append(self.add_batch_norm(node, is_training=is_training, global_norm=global_norm))
        return output_nodes
    
    def add_group_dropout_layer(self,
                                nodes,
                                keepprob):
        with tf.name_scope("GroupDropout"):
            output_nodes = list()
            for node in nodes:
                output_nodes.append(self.add_dropout_layer(node, keepprob))
        return output_nodes
        
    def add_channel_split(self,
                          node,
                          n_groups):
        if n_groups == 1:
            return [node]
        else:
            with tf.name_scope("GroupSplit"):
                nodes = list()
                
                input_channels = node.shape[3].value
                base_size = int(np.floor(input_channels / n_groups))
                leftover = input_channels - base_size * n_groups

                group_start = 0
                for group in range(n_groups):
                    group_size = base_size
                    if group < leftover:
                        group_size += 1

                    group_end = min(group_start + group_size, input_channels)
                    nodes.append(node[:, :, :, group_start:group_end])
                    group_start += group_size
            
            return nodes