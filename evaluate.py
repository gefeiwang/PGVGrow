import tensorflow as tf
import numpy as np
import scipy
import PIL
import glob
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def calculate_fid_with_act(act1, act2):
    
    mu1 = np.mean(act1, axis=0)
    mu2 = np.mean(act2, axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    sigma2 = np.cov(act2, rowvar=False)
    diff = mu1 - mu2
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(s.real)
        
def calculate_fid(data_dir1, data_dir2, batch_size, sess):

    # building graph
    raw_images = tf.placeholder(tf.float32, [None, None, None, 3], name='raw_images')
    activations1 = tf.placeholder(tf.float32, [None, None], name='activations1')
    activations2 = tf.placeholder(tf.float32, [None, None], name='activations2')
    resized_images = tf.image.resize_bilinear(raw_images, [299, 299])
    activations = tf.contrib.gan.eval.run_inception(resized_images, output_tensor = 'pool_3:0')

    # reading images
    images1, images2 = [], []
    glob_pattern1, glob_pattern2 = os.path.join(data_dir1, '*.png'), os.path.join(data_dir2, '*.png')
    image_filenames1, image_filenames2 = sorted(glob.glob(glob_pattern1)), sorted(glob.glob(glob_pattern2))
    images_num1, images_num2 = len(image_filenames1), len(image_filenames2)
    assert images_num1 == images_num2
    print('Total number of images: %d' % images_num1)
    for n in range(images_num1):
        img1 = np.asarray(PIL.Image.open(image_filenames1[n]))
        images1.append(img1)
        img2 = np.asarray(PIL.Image.open(image_filenames2[n]))
        images2.append(img2)
    images1, images2 = np.array(images1), np.array(images2)
    if len(images1.shape) == 3:
        images1 = np.tile(images1[:, :, :,np.newaxis], (1, 1, 1, 3))
        images2 = np.tile(images1[:, :, :,np.newaxis], (1, 1, 1, 3))
    assert len(images1.shape) == 4 and len(images2.shape) == 4

    # calculating
    n_batches = images_num1//batch_size
    act1, act2 = np.zeros([n_batches * batch_size, 2048], dtype = np.float32), np.zeros([n_batches * batch_size, 2048], dtype = np.float32)
    for i in tqdm(range(n_batches)):
        image_batch1 = images1[i*batch_size : (i+1)*batch_size] / 127.5 - 1
        image_batch2 = images2[i*batch_size : (i+1)*batch_size] / 127.5 - 1
        act1[i * batch_size:(i + 1) * batch_size] = sess.run(activations, feed_dict={raw_images: image_batch1})
        act2[i * batch_size:(i + 1) * batch_size] = sess.run(activations, feed_dict={raw_images: image_batch2})
    fid = calculate_fid_with_act(act1, act2)

    return fid

if __name__ == "__main__":

    tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size of images to feed into inception V3')
    tf.app.flags.DEFINE_string('path1', None, 'Image path 1')
    tf.app.flags.DEFINE_string('path2', None, 'Image path 2')
    tf.app.flags.DEFINE_string('gpu', '0', 'GPU(s) to use')
    FLAGS = tf.app.flags.FLAGS

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    sess = tf.Session()
    fid = calculate_fid(FLAGS.path1, FLAGS.path2, FLAGS.batch_size, sess)

    print(fid)