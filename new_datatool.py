import os 
import numpy as np 
import tensorflow as tf 
from glob import glob
import imageio
import argparse

def check_tfrecord_path(tfrecord_dir, dataset_name):
	# check dataset path and tfrecord file path
	
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(tfrecord_dir)==False, "You have already created this dataset!"
    os.mkdir(tfrecord_dir)
    os.mkdir(os.path.join(tfrecord_dir, 'reals-1'))
    os.mkdir(os.path.join(tfrecord_dir, 'reals-2'))

def create_tfrwriters(tfrecord_dir, dataset_name, resolution_log2):
    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join(tfrecord_dir, dataset_name + "%02d.tfrecords" % (resolution_log2 - lod)) 
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
    return tfr_writers

def create_dataset(data_dir, dataset_name, random_seed=123):
    image_types = ['*.png', '*.jpg']
    image_lists = []
    for files in image_types:
        image_lists.extend(glob(os.path.join(data_dir, files)))
    assert len(image_lists) > 0, "There is no any png or jpg file in given data path."

    tfrecord_dir = os.path.join("datasets", dataset_name)
    check_tfrecord_path(tfrecord_dir, dataset_name)

    np.random.seed(random_seed)
    np.random.shuffle(image_lists)

    # get image resolution
    image_shape = imageio.imread(image_lists[0]).shape
    resolution_log2 = int(np.log2(image_shape[1]))

    tfr_writers = create_tfrwriters(tfrecord_dir, dataset_name, resolution_log2)
    num_images = len(image_lists)

    for idx in range(num_images):
        img = imageio.imread(image_lists[idx])
        if dataset_name.lower() == "celeba":
            img = img[57:185, 25:153]
        if idx < 10000:
            imageio.imsave(os.path.join(tfrecord_dir, 'reals-1', '%06d.png' % idx), img)
        if 10000 <= idx < 20000:
            imageio.imsave(os.path.join(tfrecord_dir, 'reals-2', '%06d.png' % idx), img)
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                assert img.shape[0] == img.shape[1], "Image's width is not equal to height!"
                img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())
        if (idx + 1) % 100 == 0:
            print("Add %d / %d image \r" %(idx + 1, num_images), end='', flush=True)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Create dataset for PGVGrow')
	parser.add_argument('--image_dir', type=str, help='dir of image datasets', required=True)
	parser.add_argument('--dataset_name', type=str, help='name of image dataset', required=True)
	parser.add_argument('--random_seed', type=int, default=123, help='random seed for shuffle the image dataset')

	args = parser.parse_args()

	create_dataset(args.image_dir, args.dataset_name, args.random_seed)

