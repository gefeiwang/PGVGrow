import tensorflow as tf
import numpy as np

def lerp(a, b, t): return a + (b - a) * t

def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

def upscale2d(x, factor=2):
    s = x.shape
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
    return x

def downscale2d(x, factor=2):
    return tf.nn.avg_pool(x, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1], padding='VALID')

def pixel_norm(x, axis=3):
    return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + 1e-6)

def add_feature_stdd(x): # add standard deviation over mini-batch (one non-negetive scalar) as a new feature map
    s = x.shape
    y = tf.reduce_mean(tf.square(x - tf.reduce_mean(x, axis=0, keepdims=True)), axis=0, keepdims=True)
    y = tf.sqrt(y + 1e-6)
    y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)
    y = tf.tile(y, [tf.shape(x)[0], s[1], s[2], 1])
    return tf.concat([x, y], axis=3)

def generator(latents_in, lod_in, num_channels, resolution, latent_size, num_features, is_smoothing=False, reuse=None):

    def G_block(x, res):
        with tf.variable_scope('%dx%d' % (2**res, 2**res), reuse=tf.AUTO_REUSE):
            if res == 2:
                x = pixel_norm(x, axis=1)
                with tf.variable_scope('Dense'):
                    w = tf.get_variable('Weight', dtype=tf.float32, shape=[latent_size, num_features*16], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(2/latent_size), dtype=tf.float32, name='std')
                    x = tf.reshape(tf.matmul(x, w*std), [-1, 4, 4, num_features])
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_features], initializer=tf.initializers.zeros())
                    x = pixel_norm(tf.nn.leaky_relu(x + b, alpha=0.2), axis=1)
                with tf.variable_scope('Conv'):
                    w = tf.get_variable('Kernel', dtype=tf.float32, shape=[3, 3, num_features, num_features], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(2/(3*3*num_features)), dtype=tf.float32, name='std')
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_features], initializer=tf.initializers.zeros())
                    x = pixel_norm(tf.nn.leaky_relu(tf.nn.conv2d(x, w*std, strides=[1, 1, 1, 1], padding='SAME') + b, alpha=0.2))
            else:
                x = upscale2d(x)
                with tf.variable_scope('Conv1'):
                    w = tf.get_variable('Kernel', dtype=tf.float32, shape=[3, 3, num_features, num_features], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(2/(3*3*num_features)), dtype=tf.float32, name='std')
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_features], initializer=tf.initializers.zeros())
                    x = pixel_norm(tf.nn.leaky_relu(tf.nn.conv2d(x, w*std, strides=[1, 1, 1, 1], padding='SAME') + b, alpha=0.2))
                with tf.variable_scope('Conv2'):
                    w = tf.get_variable('Kernel', dtype=tf.float32, shape=[3, 3, num_features, num_features], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(2/(3*3*num_features)), dtype=tf.float32, name='std')
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_features], initializer=tf.initializers.zeros())
                    x = pixel_norm(tf.nn.leaky_relu(tf.nn.conv2d(x, w*std, strides=[1, 1, 1, 1], padding='SAME') + b, alpha=0.2))
            return x

    def F2I(x, res): # Feature maps to Image
        with tf.variable_scope('F2I_%dx%d' % (2**res, 2**res), reuse=tf.AUTO_REUSE):
            w = tf.get_variable('Kernel', dtype=tf.float32, shape=[1, 1, num_features, num_channels], initializer=tf.initializers.random_normal())
            std = tf.constant(np.sqrt(1/num_features), dtype=tf.float32, name='std')
            b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_channels], initializer=tf.initializers.zeros())
            return tf.nn.conv2d(x, w*std, strides=[1, 1, 1, 1], padding='SAME') + b

    def grow(x, res, lod):
        y = G_block(x, res)
        img = lambda: upscale2d(F2I(y, res), 2**lod)
        if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(F2I(y, res), upscale2d(F2I(x, res - 1)), lod_in - lod), 2**lod))
        if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
        return img()

    scope = 'generator_smoothing' if is_smoothing else 'generator'
    with tf.variable_scope(scope, reuse=reuse):
        resolution_log2 = int(np.log2(resolution))
        images_out = grow(latents_in, 2, resolution_log2 - 2)
        return images_out

def discriminator(images_in, lod_in, num_channels, resolution, num_features, reuse=None):

    def D_block(x, res):
        with tf.variable_scope('%dx%d' % (2**res, 2**res), reuse=tf.AUTO_REUSE):
            if res >= 3:
                with tf.variable_scope('Conv1'):
                    w = tf.get_variable('Kernel', dtype=tf.float32, shape=[3, 3, num_features, num_features], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(2/(3*3*num_features)), dtype=tf.float32, name='std')
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_features], initializer=tf.initializers.zeros())
                    x = tf.nn.leaky_relu(tf.nn.conv2d(x, w*std, strides=[1, 1, 1, 1], padding='SAME') + b, alpha=0.2)
                with tf.variable_scope('Conv2'):
                    w = tf.get_variable('Kernel', dtype=tf.float32, shape=[3, 3, num_features, num_features], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(2/(3*3*num_features)), dtype=tf.float32, name='std')
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_features], initializer=tf.initializers.zeros())
                    x = tf.nn.leaky_relu(tf.nn.conv2d(x, w*std, strides=[1, 1, 1, 1], padding='SAME') + b, alpha=0.2)
                x = downscale2d(x)
            else:
                x = add_feature_stdd(x)
                with tf.variable_scope('Conv'):
                    w = tf.get_variable('Kernel', dtype=tf.float32, shape=[3, 3, num_features+1, num_features], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(2/(3*3*num_features)), dtype=tf.float32, name='std')
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_features], initializer=tf.initializers.zeros())
                    x = tf.nn.leaky_relu(tf.nn.conv2d(x, w*std, strides=[1, 1, 1, 1], padding='SAME') + b, alpha=0.2)
                with tf.variable_scope('Dense1'):
                    x = tf.layers.flatten(x)
                    w = tf.get_variable('Weight', dtype=tf.float32, shape=[16*num_features, num_features], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(2/(16*num_features)), dtype=tf.float32, name='std')
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=num_features, initializer=tf.initializers.zeros())
                    x = tf.nn.leaky_relu(tf.nn.xw_plus_b(x, w*std, b), alpha=0.2)
                with tf.variable_scope('Dense2'):
                    w = tf.get_variable('Weight', dtype=tf.float32, shape=[num_features, 1], initializer=tf.initializers.random_normal())
                    std = tf.constant(np.sqrt(1/num_features), dtype=tf.float32, name='std')
                    b = tf.get_variable('Bias', dtype=tf.float32, shape=1, initializer=tf.initializers.zeros())
                    x = tf.nn.xw_plus_b(x, w*std, b)
            return x

    def I2F(x, res): # Image to Feature maps
        with tf.variable_scope('I2F_%dx%d' % (2**res, 2**res), reuse=tf.AUTO_REUSE):
            w = tf.get_variable('Kernel', dtype=tf.float32, shape=[1, 1, num_channels, num_features], initializer=tf.initializers.random_normal())
            std = tf.constant(np.sqrt(2/num_channels), dtype=tf.float32, name='std')
            b = tf.get_variable('Bias', dtype=tf.float32, shape=[num_features], initializer=tf.initializers.zeros())
            return tf.nn.leaky_relu(tf.nn.conv2d(x, w*std, strides=[1, 1, 1, 1], padding='SAME') + b, alpha=0.2)

    def grow(res, lod):
        x = lambda: I2F(downscale2d(images_in, 2**lod), res)
        if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
        x = D_block(x(), res); y = lambda: x
        if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, I2F(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
        return y()

    with tf.variable_scope('discriminator', reuse=reuse):
        resolution_log2 = int(np.log2(resolution))
        scores_out = grow(2, resolution_log2 - 2)
        return scores_out
