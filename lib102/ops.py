# uncompyle6 version 3.3.5
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.1 (v3.7.1:260ec2c36a, Oct 20 2018, 14:57:15) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: lib\ops.py
import tensorflow as tf, tensorflow.contrib.slim as slim, pdb, keras, numpy as np, cv2 as cv, scipy, collections

def preprocess(image):
    with tf.name_scope('preprocess'):
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope('deprocess'):
        return (image + 1) / 2


def preprocessLR(image):
    with tf.name_scope('preprocessLR'):
        return tf.identity(image)


def deprocessLR(image):
    with tf.name_scope('deprocessLR'):
        return tf.identity(image)


def conv2_tran(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC', activation_fn=None,
              weights_initializer=(tf.contrib.layers.xavier_initializer()))
        else:
            return slim.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC', activation_fn=None,
              weights_initializer=(tf.contrib.layers.xavier_initializer()),
              biases_initializer=None)


def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC', activation_fn=None,
              weights_initializer=(tf.contrib.layers.xavier_initializer()))
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC', activation_fn=None,
              weights_initializer=(tf.contrib.layers.xavier_initializer()),
              biases_initializer=None)


def conv2_NCHW(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv_NCHW'):
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NCWH', activation_fn=None,
              weights_initializer=(tf.contrib.layers.xavier_initializer()))
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NCWH', activation_fn=None,
              weights_initializer=(tf.contrib.layers.xavier_initializer()),
              biases_initializer=None)


def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', (inputs.get_shape()[(-1)]), initializer=(tf.zeros_initializer()), collections=[
         tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.MODEL_VARIABLES],
          dtype=(tf.float32))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def lrelu(inputs, alpha):
    return keras.layers.LeakyReLU(alpha=alpha).call(inputs)


def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=(tf.GraphKeys.UPDATE_OPS), scale=False,
      fused=True,
      is_training=is_training)


def maxpool(inputs, scope='maxpool'):
    return slim.max_pool2d(inputs, [2, 2], scope=scope)


def denselayer(inputs, output_size):
    denseLayer = tf.layers.Dense(output_size, activation=None, kernel_initializer=(tf.contrib.layers.xavier_initializer()))
    output = denseLayer.apply(inputs)
    tf.add_to_collection(name=(tf.GraphKeys.MODEL_VARIABLES), value=(denseLayer.kernel))
    return output


def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[(-1)]
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target
    shape_1 = [
     batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)
    return output


def upscale_four(inputs, scope='upscale_four'):
    with tf.variable_scope(scope):
        size = tf.shape(inputs)
        b = size[0]
        h = size[1]
        w = size[2]
        c = size[3]
        p_inputs = tf.concat((inputs, inputs[:, -1:, :, :]), axis=1)
        p_inputs = tf.concat((p_inputs, p_inputs[:, :, -1:, :]), axis=2)
        hi_res_bin = [
         [
          inputs,
          p_inputs[:, :-1, 1:, :]],
         [
          p_inputs[:, 1:, :-1, :],
          p_inputs[:, 1:, 1:, :]]]
        hi_res_array = []
        for hi in range(4):
            for wj in range(4):
                hi_res_array.append(hi_res_bin[0][0] * (1.0 - 0.25 * hi) * (1.0 - 0.25 * wj) + hi_res_bin[0][1] * (1.0 - 0.25 * hi) * (0.25 * wj) + hi_res_bin[1][0] * (0.25 * hi) * (1.0 - 0.25 * wj) + hi_res_bin[1][1] * (0.25 * hi) * (0.25 * wj))

        hi_res = tf.stack(hi_res_array, axis=3)
        hi_res_reshape = tf.reshape(hi_res, (b, h, w, 4, 4, c))
        hi_res_reshape = tf.transpose(hi_res_reshape, (0, 1, 3, 2, 4, 5))
        hi_res_reshape = tf.reshape(hi_res_reshape, (b, h * 4, w * 4, c))
    return hi_res_reshape


def phaseShift(inputs, scale, shape_1, shape_2):
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])
    return tf.reshape(X, shape_2)


def random_flip(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.cond(tf.less(decision, 0.5), lambda : f2, lambda : f1)
    return output


def print_configuration_op(FLAGS):
    print('[Configurations]:')
    for name, value in FLAGS.flag_values_dict().items():
        print('\t%s: %s' % (name, str(value)))

    print('End of configuration')


def copy_update_configuration(FLAGS, updateDict={}):
    namelist = []
    valuelist = []
    for name, value in FLAGS.flag_values_dict().items():
        namelist += [name]
        if name in updateDict:
            valuelist += [updateDict[name]]
        else:
            valuelist += [value]

    Params = collections.namedtuple('Params', ','.join(namelist))
    tmpFLAGS = Params._make(valuelist)
    return tmpFLAGS


def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10.0 * (tf.log(65025.0 / mse) / tf.log(10.0))
    return psnr


def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=(tf.nn.relu),
      weights_regularizer=(slim.l2_regularizer(weight_decay)),
      biases_initializer=(tf.zeros_initializer())):
        with slim.arg_scope([slim.conv2d], padding='SAME') as (arg_sc):
            return arg_sc


def vgg_19(inputs, num_classes=1000, is_training=False, dropout_keep_prob=0.5, spatial_squeeze=True, scope='vgg_19', reuse=False, fc_conv_padding='VALID'):
    """Oxford Net VGG 19-Layers version E Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as (sc):
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, (slim.conv2d), 64, 3, scope='conv1', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, (slim.conv2d), 128, 3, scope='conv2', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, (slim.conv2d), 256, 3, scope='conv3', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, (slim.conv2d), 512, 3, scope='conv4', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, (slim.conv2d), 512, 3, scope='conv5', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return (
             net, end_points)


vgg_19.default_image_size = 224

def gaussian_2dkernel(size=5, sig=1.0):
    """
    Returns a 2D Gaussian kernel array with side length size and a sigma of sig
    """
    gkern1d = signal.gaussian(size, std=sig).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


def tf_data_gaussDownby4(HRdata, sigma=1.5):
    """
    tensorflow version of the 2D down-scaling by 4 with Gaussian blur
    sigma: the sigma used for Gaussian blur
    return: down-scaled data
    """
    k_w = 1 + 2 * int(sigma * 3.0)
    gau_k = gaussian_2dkernel(k_w, sigma)
    gau_0 = np.zeros_like(gau_k)
    gau_list = np.float32([
     [
      gau_k, gau_0, gau_0],
     [
      gau_0, gau_k, gau_0],
     [
      gau_0, gau_0, gau_k]])
    gau_wei = np.transpose(gau_list, [2, 3, 0, 1])
    with tf.device('/gpu:0'):
        fix_gkern = tf.constant(gau_wei, dtype=(tf.float32), shape=[k_w, k_w, 3, 3], name='gauss_blurWeights')
        cur_data = tf.nn.conv2d(HRdata, fix_gkern, strides=[1, 4, 4, 1], padding='VALID', name='gauss_downsample_4')
        return cur_data


def warp_flow(img, flowo):
    h, w = flowo.shape[:2]
    flow = -flowo
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    ih, iw = img.shape[:2]
    flow[:, :, 0] = np.clip(flow[:, :, 0], 0, iw)
    flow[:, :, 1] = np.clip(flow[:, :, 1], 0, ih)
    res = cv.remap(img, np.float32(flow), None, cv.INTER_LINEAR)
    return res


def warp_flow_rz(img, vdata, up4=True):
    b = 1
    h = vdata.shape[0]
    w = vdata.shape[1]
    c = 2
    if up4:
        p_inputs = np.concatenate((vdata, vdata[-1:, :, :]), axis=0)
        p_inputs = np.concatenate((p_inputs, p_inputs[:, -1:, :]), axis=1)
        hi_res_bin = [
         [
          vdata,
          p_inputs[:-1, 1:, :]],
         [
          p_inputs[1:, :-1, :],
          p_inputs[1:, 1:, :]]]
        hi_res_array = []
        for hi in range(4):
            for wj in range(4):
                hi_res_array.append(hi_res_bin[0][0] * (1.0 - 0.25 * hi) * (1.0 - 0.25 * wj) + hi_res_bin[0][1] * (1.0 - 0.25 * hi) * (0.25 * wj) + hi_res_bin[1][0] * (0.25 * hi) * (1.0 - 0.25 * wj) + hi_res_bin[1][1] * (0.25 * hi) * (0.25 * wj))

        hi_res = np.stack(hi_res_array, axis=2)
        hi_res_reshape = np.reshape(hi_res, (h, w, 4, 4, c))
        hi_res_reshape = np.transpose(hi_res_reshape, (0, 2, 1, 3, 4))
        hi_res_reshape = np.reshape(hi_res_reshape, (h * 4, w * 4, c))
    else:
        hi_res_reshape = vdata
    ih, iw = img.shape[:2]
    if ih > hi_res_reshape.shape[0]:
        hi_res_reshape = np.pad(hi_res_reshape, ((0, ih - hi_res_reshape.shape[0]), (0, 0), (0, 0)), 'linear_ramp')
    if iw > hi_res_reshape.shape[1]:
        hi_res_reshape = np.pad(hi_res_reshape, ((0, 0), (0, iw - hi_res_reshape.shape[1]), (0, 0)), 'linear_ramp')
    data2 = hi_res_reshape
    return warp_flow(img, data2)


def save_img(out_path, img):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.shape[(-1)] == 2:
        img = np.stack((img[:, :, 0], img[:, :, 1], 0.5 * (img[:, :, 0] + img[:, :, 1])), axis=(-1))
    cv.imwrite(out_path, img)