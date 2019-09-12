# uncompyle6 version 3.3.5
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.1 (v3.7.1:260ec2c36a, Oct 20 2018, 14:57:15) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: lib\frvsr.py
from lib.dataloader import *

def fnet(fnet_input, reuse=False):

    def down_block(inputs, output_channel=64, stride=1, scope='down_block'):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
            net = lrelu(net, 0.2)
            net = conv2(net, 3, output_channel, stride, use_bias=True, scope='conv_2')
            net = lrelu(net, 0.2)
            net = maxpool(net)
        return net

    def up_block(inputs, output_channel=64, stride=1, scope='up_block'):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
            net = lrelu(net, 0.2)
            net = conv2(net, 3, output_channel, stride, use_bias=True, scope='conv_2')
            net = lrelu(net, 0.2)
            new_shape = tf.shape(net)[1:-1] * 2
            net = tf.image.resize_images(net, new_shape)
        return net

    with tf.variable_scope('autoencode_unit', reuse=reuse):
        net = down_block(fnet_input, 32, scope='encoder_1')
        net = down_block(net, 64, scope='encoder_2')
        net = down_block(net, 128, scope='encoder_3')
        net = up_block(net, 256, scope='decoder_1')
        net = up_block(net, 128, scope='decoder_2')
        net1 = up_block(net, 64, scope='decoder_3')
        with tf.variable_scope('output_stage'):
            net = conv2(net1, 3, 32, 1, scope='conv1')
            net = lrelu(net, 0.2)
            net2 = conv2(net, 3, 2, 1, scope='conv2')
            net = tf.tanh(net2) * 24.0
    return net


def generator_F(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    def residual_block(inputs, output_channel=64, stride=1, scope='res_block'):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
            net = tf.nn.relu(net)
            net = conv2(net, 3, output_channel, stride, use_bias=True, scope='conv_2')
            net = net + inputs
        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 3, 64, 1, scope='conv')
            stage1_output = tf.nn.relu(net)
        net = stage1_output
        for i in range(1, FLAGS.num_resblock + 1, 1):
            name_scope = 'resblock_%d' % i
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('conv_tran2highres'):
            net = conv2_tran(net, 3, 64, 2, scope='conv_tran1')
            net = tf.nn.relu(net)
            net = conv2_tran(net, 3, 64, 2, scope='conv_tran2')
            net = tf.nn.relu(net)
        with tf.variable_scope('output_stage'):
            net = conv2(net, 3, gen_output_channels, 1, scope='conv')
            hi_shape = tf.shape(net)
            low_res_in = gen_inputs[:, :, :, 0:3]
            bicubic_hi = tf.image.resize_bicubic(low_res_in, (hi_shape[1], hi_shape[2]))
            net = net + bicubic_hi
            net = preprocess(net)
    return net