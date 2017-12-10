from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import tensorflow.contrib.layers as L
import tensorflow.contrib.slim as slim


# credit Laurnet Mazare 
def batch_activ_conv(input_tensor, num_outputs,  kernel_size, is_training, keep_prob, scope):
    
    fn_conv = slim.conv3d
    num_inputs  = input_tensor.get_shape()[-1].value
    with tf.variable_scope(scope):
        net = L.batch_norm(input_tensor,
                           scale=True, 
                           is_training=is_training, 
                           scope=scope)
        net = fn_conv(inputs = net,
                      num_outputs = num_outputs,
                      kernel_size = kernel_size,
                      stride = 1,
                      trainable = is_training,
                      normalizer_fn = None,
                      activation_fn = tf.nn.relu,
                      scope = scope)
        net = tf.nn.dropout(net, keep_prob)
    return net

def block(input_tensor, num_layers, growth_per_layer, is_training=True, keep_prob=1., scope='block_scope'):
    
    current = input_tensor

    with tf.variable_scope(scope):
        for i in xrange(num_layers):
            tmp = batch_activ_conv(input_tensor = current, 
                                   num_outputs = growth_per_layer,
                                   kernel_size = 3,
                                   is_training = is_training, 
                                   keep_prob = keep_prob, 
                                   scope = 'layer_'+str(i))
            current = tf.concat((current, tmp), axis=len(input_tensor.get_shape())-1 )
            
    return current


def transition_down(input_tensor, is_training, scope='transition_scope'):

    fn_conv = slim.conv3d
    num_inputs  = input_tensor.get_shape()[-1].value

    with tf.variable_scope(scope):
        net = L.batch_norm(input_tensor,
                           scale=True, 
                           is_training=is_training,
                           scope=scope)
        net = fn_conv(inputs = net,
                      num_outputs = num_inputs,
                      kernel_size = 1,
                      stride = 1,
                      trainable = is_training,
                      normalizer_fn = None,
                      activation_fn = tf.nn.relu,
                      scope = scope)
        net = L.avg_pool3d(inputs = net,
                           kernel_size = 2,
                           stride = 2,
                           padding = 'valid')

    return net

    
def transition_up(shortcut, to_concat, is_training, num_outputs, scope):
    
    fn_deconv = slim.conv3d_transpose

    net = tf.concat(to_concat, axis=len(to_concat[0].get_shape())-1)
    net = fn_deconv(inputs = net,
                    num_outputs = num_outputs,
                    kernel_size = 3,
                    stride = 2,
                    padding = 'same', #### double check
                    activation_fn = None,
                    trainable = is_training,
                    scope = scope)
    net = tf.concat([net, shortcut], axis=len(net.get_shape())-1)

    return net
    

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50, 128,128,128,1])
    net = block(input_tensor=x, num_layers=4, growth_per_layer=32, scope='test_block')
    print(net.get_shape())
    net = transition_down(input_tensor = net, is_training = True, scope='test_transition_down')
    print(net.get_shape())

    y = tf.placeholder(tf.float32, [50, 128, 128, 128, 5])
    net = transition_up(shortcut = y, to_concat = [net, net], is_training = True, num_outputs = 32, scope = 'test_transition_up')
    print(net.get_shape())

    import sys
    if 'save' in sys.argv:
        # Create a session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        # Create a summary writer handle + save graph
        writer=tf.summary.FileWriter('double_resnet_graph')
        writer.add_graph(sess.graph)
