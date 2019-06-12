import tensorflow as tf 
import os
import numpy as np 
from glob import glob
from utils import *

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

class Dataset(object):
    def __init__(self, tfrecord_path, number_thread=2, buffer_mb=256, prefetch_mb=2048, shuffle_mb = 4096):
        self.tfrecord_path = tfrecord_path

        self.number_thread = number_thread
        self.buffer_mb = buffer_mb
        self.prefetch_mb = prefetch_mb
        self.shuffle_mb = shuffle_mb

        self.previous_batch_size = 0
        self.previous_resolution = 0

        self.multiresolution_iterator = dict()

        self.previous_batch_size      = -1
        self.previous_resolution_log2 = -1

        self.build_graph()

    def build_graph(self):
        with tf.name_scope("Dataset"), tf.device('/cpu:0'):
            self._batch_size = tf.placeholder(tf.int64, name="current_batch_size", shape=[])
            assert os.path.isdir(self.tfrecord_path), "Wrong path to tfrecords files"
            self.tfrecord_files = sorted(glob(os.path.join(self.tfrecord_path, "*.tfrecords")))
            assert len(self.tfrecord_files) >= 1, "There is no tfrecords file in your path!"
            self._shape_inference()
            for shape, record_file in zip(self.tfr_shapes, self.tfrecord_files):
                dset = tf.data.TFRecordDataset(record_file, compression_type='', buffer_size=self.buffer_mb<<20)
                dset = dset.map(parse_tfrecord_tf, num_parallel_calls=self.number_thread)
                bytes_per_image = np.prod(shape) * np.dtype("uint8").itemsize
                dset = dset.shuffle(((self.shuffle_mb << 20) - 1) // bytes_per_image + 1)
                dset = dset.repeat()
                dset = dset.prefetch(((self.prefetch_mb << 20) - 1) // bytes_per_image + 1)
                dset = dset.batch(self._batch_size)
                index = np.log2(shape[1])
                self.multiresolution_iterator[index] = dset
            self._tf_iterator = tf.data.Iterator.from_structure(self.multiresolution_iterator[self.resolution_log2].output_types, self.multiresolution_iterator[self.resolution_log2].output_shapes)
            self._tf_init_ops = {index: self._tf_iterator.make_initializer(dset) for index, dset in self.multiresolution_iterator.items()}

    def _shape_inference(self):
        self.tfr_shapes = []
        for tfr_file in self.tfrecord_files:
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
                self.tfr_shapes.append(parse_tfrecord_np(record).shape)
                break
        self.max_shape = max(self.tfr_shapes, key=lambda shape: np.prod(shape))
        assert self.max_shape[-1] in [1, 3], "Wrong image format!"
        self.resolution_log2 = np.log2(self.max_shape[0])

    def set_information(self):
        resolution = self.max_shape[1]
        num_channels = self.max_shape[-1]
        if resolution <= 128:
            num_features = 256 
        else:
            num_features = 128
        return num_channels, resolution, num_features

    def configure(self, lod_in, minibatch_size):
        curr_resolution_log2 = self.resolution_log2 - np.floor(lod_in)
        assert minibatch_size >= 1 and lod_in >= 0
        if self.previous_batch_size != minibatch_size or self.previous_resolution_log2 != curr_resolution_log2:
            self._tf_init_ops[curr_resolution_log2].run({self._batch_size: minibatch_size})
            self.previous_batch_size = minibatch_size
            self.previous_resolution_log2 = curr_resolution_log2

    def get_minibatch(self):
        return self._tf_iterator.get_next()