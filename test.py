import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import misc
from wssdcnn import model


def test():
    inp = tf.placeholder(tf.float32, [None, 224, None, 3])
    ar = tf.placeholder(tf.float32, [])
    m = model()
    res, a_map = m.test(inp, ar)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'model_ckpt/model.ckpt')

    input_dir = 'input_dir'
    output_dir = 'output_dir'
    r = 0.5  # aspect ratio

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_names = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    for img_name in img_names:
        print(img_name)
        img = misc.imread(os.path.join(input_dir, img_name))
        h, w, _ = img.shape
        scale = 224.0 / h
        h = 224
        w = int(w * scale)
        img = misc.imresize(img, [h, w])
        p_name = os.path.splitext(img_name)[0]
        misc.imsave(os.path.join(output_dir, img_name), img)

        tar_w = int(w * r)
        resize_img = misc.imresize(img, [224, tar_w])
        misc.imsave(os.path.join(output_dir, p_name + '_bi.png'), resize_img)

        img = np.expand_dims(img, 0).astype(np.float32)
        out = sess.run(res, feed_dict={inp: img, ar: r})
        out = np.round(np.clip(out[0], 0, 255)).astype(np.uint8)
        misc.imsave(os.path.join(output_dir, p_name + '_%s.png' % str(r)), out)


if __name__ == "__main__":
    test()
