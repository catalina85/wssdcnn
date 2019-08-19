import tensorflow as tf
from vgg16 import our_vgg_16

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def conv2d(x, out_number, name, stride=1, kernel_size=3, act=tf.nn.elu):
    return tf.layers.conv2d(x, out_number, kernel_size, stride, 'same', activation=act, name=name,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-5))


def cumulative_norm(x):
    y = tf.reduce_sum(x, axis=2, keepdims=True)
    x = tf.cumsum(x, axis=2)
    return x / y


def tf_inverse_warp(input, flow):
    '''
    linear interpolation for Image Warping
    '''
    shape = tf.shape(input)
    i_H = shape[1]
    i_W = shape[2]
    shape = tf.shape(flow)
    N = shape[0]
    H = shape[1]
    W = shape[2]

    N_i = tf.range(0, N)
    W_i = tf.range(0, W)
    H_i = tf.range(0, H)

    n, h, w = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = tf.expand_dims(n, axis=3)
    h = tf.expand_dims(h, axis=3)
    w = tf.expand_dims(w, axis=3)

    n = tf.cast(n, tf.float32)
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)

    v_col, v_row = tf.split(flow, 2, axis=-1)

    v_r0 = tf.floor(v_row)
    v_r1 = v_r0 + 1
    v_c0 = tf.floor(v_col)
    v_c1 = v_c0 + 1

    H_ = tf.cast(i_H - 1, tf.float32)
    W_ = tf.cast(i_W - 1, tf.float32)
    i_r0 = tf.clip_by_value(h + v_r0, 0., H_)
    i_r1 = tf.clip_by_value(h + v_r1, 0., H_)
    i_c0 = tf.clip_by_value(w + v_c0, 0., W_)
    i_c1 = tf.clip_by_value(w + v_c1, 0., W_)

    i_r0c0 = tf.cast(tf.concat([n, i_r0, i_c0], axis=-1), tf.int32)
    i_r0c1 = tf.cast(tf.concat([n, i_r0, i_c1], axis=-1), tf.int32)
    i_r1c0 = tf.cast(tf.concat([n, i_r1, i_c0], axis=-1), tf.int32)
    i_r1c1 = tf.cast(tf.concat([n, i_r1, i_c1], axis=-1), tf.int32)

    f00 = tf.gather_nd(input, i_r0c0)
    f01 = tf.gather_nd(input, i_r0c1)
    f10 = tf.gather_nd(input, i_r1c0)
    f11 = tf.gather_nd(input, i_r1c1)

    w00 = (v_r1 - v_row) * (v_c1 - v_col)
    w01 = (v_r1 - v_row) * (v_col - v_c0)
    w10 = (v_row - v_r0) * (v_c1 - v_col)
    w11 = (v_row - v_r0) * (v_col - v_c0)

    out = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    return out


class model():
    def build(self, x, aspect_radio, inp_h=224):
        img = x
        b, h, w, c = tf.unstack(tf.shape(x))
        x = x - [_R_MEAN, _G_MEAN, _B_MEAN]
        vgg_logits, end_points = our_vgg_16(x, 20, is_training=False)
        encoder = end_points['vgg_16/conv5/conv5_3']
        tar_w = tf.cast(tf.cast(w, tf.float32) * aspect_radio, tf.int32)
        with tf.variable_scope('retargeting', reuse=tf.AUTO_REUSE):
            x = conv2d(encoder, 512, 'conv5_0')
            x = conv2d(x, 512, 'conv5_1')
            x = conv2d(x, 512, 'conv5_2')
            x = tf.image.resize_bilinear(x, tf.shape(x)[1:3] * 2)
            x = conv2d(x, 512, 'conv4_0')
            x = conv2d(x, 512, 'conv4_1')
            x = conv2d(x, 512, 'conv4_2')
            x = tf.image.resize_bilinear(x, tf.shape(x)[1:3] * 2)
            x = conv2d(x, 256, 'conv3_0')
            x = conv2d(x, 256, 'conv3_1')
            x = conv2d(x, 256, 'conv3_2')
            x = tf.image.resize_bilinear(x, tf.shape(x)[1:3] * 2)
            x = conv2d(x, 128, 'conv2_0')
            x = conv2d(x, 128, 'conv2_1')
            x = tf.image.resize_bilinear(x, tf.shape(x)[1:3] * 2)
            x = conv2d(x, 64, 'conv1_0')
            x = conv2d(x, 1, 'conv1_1', act=None)
            a1 = tf.image.resize_bilinear(x, [inp_h, tar_w])
            a2 = tf.layers.conv2d(a1, 1, [inp_h, 1], padding='valid', activation=None)
            a2 = tf.tile(a2, [1, inp_h, 1, 1])
            a = a1 + 0.3 * a2
            self.attention_map = a
            a = tf.cast(tf.abs(w - tar_w), tf.float32) * cumulative_norm(a)
            flow = tf.concat([a, tf.zeros(tf.shape(a))], -1)
            img = tf_inverse_warp(img, flow)
            return img, flow, end_points, self.attention_map

    def train(self, x, aspect_radio, gt):
        img, flow, x_points, attention_map = self.build(x, aspect_radio)

        self.bilinear = tf.image.resize_bilinear(x, tf.shape(img)[1:3])
        vgg_inp = img
        w = tf.shape(img)[2]
        dw = tf.shape(x)[2] - w
        vgg_inp = tf.pad(vgg_inp, [[0, 0], [0, 0], [0, dw], [0, 0]])
        vgg_inp = tf.reshape(vgg_inp, [-1, 224, 224, 3])
        self.vgg_inp = vgg_inp
        vgg_inp = vgg_inp - [_R_MEAN, _G_MEAN, _B_MEAN]
        i_logits, i_points = our_vgg_16(vgg_inp, 20, is_training=False)
        i_logits = tf.sigmoid(i_logits)
        i_conv1_1 = i_points['vgg_16/conv1/conv1_1'][:, :, :w]
        i_conv1_2 = i_points['vgg_16/conv1/conv1_2'][:, :, :w]
        content_loss = -1.0 * tf.reduce_mean(gt * tf.log(i_logits + 1e-8) + (1 - gt) * tf.log(1 - i_logits + 1e-8))
        self.error_rate = tf.reduce_mean(tf.reduce_sum((gt - i_logits) * gt, -1) / tf.reduce_sum(gt, -1))

        # for computing Structure Loss
        x_conv1_1 = x_points['vgg_16/conv1/conv1_1']
        x_conv1_2 = x_points['vgg_16/conv1/conv1_2']

        x_conv1_1 = tf_inverse_warp(x_conv1_1, flow)
        x_conv1_2 = tf_inverse_warp(x_conv1_2, flow)
        structure_loss = (tf.reduce_mean(tf.abs(i_conv1_1 - x_conv1_1)) + tf.reduce_mean(
            tf.abs(i_conv1_2 - x_conv1_2))) * 0.001
        return img, flow, content_loss, structure_loss

    def test(self, x, aspect_radio):
        img, flow, x_points, a = self.build(x, aspect_radio)
        return img, a
