from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import tensorflow.contrib.layers as L
import tensorflow.contrib.slim as slim
from denseblock import block, transition_down, batch_activ_conv, transition_up
from ssnet import ssnet_base

class fcdensenet(ssnet_base):

    def __init__(self, dims, num_class, num_down, num_layers, num_filters_base, growth, keep_prob, num_strides=5, debug=False):
        super(fcdensenet,self).__init__(dims=dims,
                                        num_class=num_class)
        self._num_filters_base = int(num_filters_base)
        self._num_strides = int(num_strides)
        self._debug = bool(debug)
        self._num_layers = num_layers
        self._growth = int(growth)
        self._num_down = int(num_down)
        self._keep_prob = keep_prob

    def _build(self,input_tensor):
        
        if self._debug: print(input_tensor.shape, 'input shape')

        # len(input_tensor.shape) == 5:
        fn_conv = slim.conv3d
        fn_conv_transpose = slim.conv3d_transpose

        with tf.variable_scope('FCDenseNet'):

            #
            # DOWNSAMPLING PATH
            #

            shortcuts = []
            num_filters = 0

            # initial convolution
            net = fn_conv(inputs      = input_tensor,
                          num_outputs = self._num_filters_base,
                          kernel_size = 3,
                          stride      = 1,
                          trainable   = self._trainable,
                          normalizer_fn = None,
                          activation_fn = None,
                          padding     = 'same',
                          scope       = 'conv0')
            num_filters = self._num_filters_base
            if self._debug: print(net.shape, 'after conv0')
                        
            for block_i in xrange(self._num_down):
                with tf.variable_scope('block_' + str(block_i)):
                    net = block(input_tensor = net,
                                num_layers = self._num_layers[block_i],
                                growth_per_layer = self._growth,
                                is_training = self._trainable,
                                keep_prob = self._keep_prob,
                                scope = 'downblock_'+str(block_i))
                    num_filters += self._num_layers[block_i]*self._growth
                    shortcuts.append(net)
                    net = transition_down(input_tensor = net,
                                          is_training = self._trainable,
                                          scope = 'downtransition_'+str(block_i))
                if self._debug: print(net.shape, 'after down block'+str(block_i)))

            shortcuts = shortcuts[::-1]
            
            #
            # FINAL DOWN-SAMPLING DENSE BLOCK
            #

            to_upsample = []
A1;95;0c
            for layer_i in xrange(self._num_layers[num_down]):
                temp = batch_activ_conv(net,
                                        num_outputs = self._growth,
                                        kernel_size = 3,
                                        is_training = self._trainable,
                                        keep_prob = self._keep_prob, 
                                        scope = 'batch_activ_conv_middle')
                to_upsample.append(temp)
                net = tf.concat([net, temp],
                                axis=len(net.shape)-1,
                                name='middle_concat%d' % layer_i)
            if self._debug: print(net.shape, 'after final downblock')
            #
            # UPSAMPLING PATH
            #

            for block_i in xrange(num_down): # note num_down = num_up
                n_filters_up = self._growth*self._num_layers[num_down + block_i]
                net = transition_up(shortcut = shortcuts[block_i], 
                                    to_concat = to_upsample,
                                    num_outputs = n_filters_up
                                    is_training = self._trainable,
                                    scope = 'uptransition_%d' %block_i)
             
                # dense block, while we update to_upsample
                to_upsample = []
                for layer_i in xrange(self._num_layers[num_down]):
                    temp = batch_activ_conv(net,
                                            num_outputs = self._growth,
                                            kernel_size = 3,
                                            is_training = self._trainable,
                                            keep_prob = self._keep_prob,
                                            scope = 'batch_activ_conv_up%d' %block_i)
                    to_upsample.append(temp)
                    net = tf.concat([net, temp],
                                    axis=len(net.shape)-1,
                                    name='up%d_concat_layer%d' % (block_i, layer_i))
                if self._debug: print(net.shape, 'after up block%d' block_i))

            return net


if __name__ == '__main__':

    import sys
    dims = [512,512,1]
    if '3d' in sys.argv:
        dims = [128,128,128,1]
    # some constants
    BATCH=1
    NUM_CLASS=3
    # make network
    net = uresnet(dims=dims,
                  num_class=NUM_CLASS,
                  debug=True)
    net.construct(trainable=True,use_weight=True)

    import sys
    if 'save' in sys.argv:
        # Create a session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        # Create a summary writer handle + save graph
        writer=tf.summary.FileWriter('uresnet_graph')
        writer.add_graph(sess.graph)
