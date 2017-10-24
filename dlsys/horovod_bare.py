from __future__ import print_function
import itertools
import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()
rank = hvd.rank()
size = hvd.size()
print('rank', rank, 'of', size)

with tf.Session() as sess:
    dtypes = [tf.int32, tf.int64, tf.float32, tf.float64]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        with tf.device("/cpu:0"):
            tf.set_random_seed(1234)
            tensor = tf.random_uniform(
                [17] * dim, -100, 100, dtype=dtype)
            summed = hvd.allreduce(tensor, average=False)
        multiplied = tensor * size
        max_difference = tf.reduce_max(tf.abs(summed - multiplied))

        diff = sess.run(max_difference)
        print(diff)
