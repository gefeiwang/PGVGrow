import tensorflow as tf
import numpy as np

################################ visualizing image grids ################################

def montage(images, grid):

    s = np.shape(images)
    assert s[0] == np.prod(grid) and np.shape(s)[0] == 4
    bigimg = np.zeros((s[1]*grid[0], s[1]*grid[1], s[3]), dtype=np.float32)

    for i in range(grid[0]):
        for j in range(grid[1]):
            bigimg[s[1] * i : s[1] * i + s[1], s[1] * j : s[1] * j + s[1]] += images[grid[1] * i + j]

    return np.rint(bigimg*255).clip(0, 255).astype(np.uint8)

################################ pre-processing real images ################################

def downscale(img):
    s = tf.shape(img)
    y = tf.reshape(img, [-1, s[1]//2, 2, s[2]//2, 2, s[3]])
    return tf.reduce_mean(y, axis=[2, 4])

def upscale(img, factor):
    s = tf.shape(img)
    y = tf.reshape(img, [-1, s[1], 1, s[2], 1, s[3]])
    y = tf.tile(y, [1, 1, factor, 1, factor, 1])
    return tf.reshape(y, [-1, s[1]*factor, s[2]*factor, s[3]])

def process_real(x, lod_in):
    with tf.name_scope('process_real'):
        x = tf.cast(x, tf.float32)
        y = x / 127.5 - 1
        alpha = lod_in - tf.floor(lod_in)
        y = (1 - alpha)*y + alpha*upscale(downscale(y), factor=2)
        factor = tf.cast(2 ** tf.floor(lod_in), tf.int32)
        y = upscale(y, factor)
        return y