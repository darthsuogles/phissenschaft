"""
To use PySpark, eval (setq python-shell-interpreter "./.ipy3")
"""
import tensorflow as tf
from tensorflow import keras
Lyr = keras.layers

# Create a spark session
try:
    print(spark)
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[4]").appName("tnsrflw").getOrCreate()

ds1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(ds1)

ds2 = tf.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]),
     tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(ds2)

ds3 = tf.data.Dataset.zip((ds1, ds2))
print(ds3)

# Setup graph and session for interactive development
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
tf.keras.backend.set_session(sess)

# Build a simple model
model = keras.models.Sequential()
# |Vocab| = 1000 => R^{64}
model.add(Lyr.Embedding(1000, 64, input_length=80))
model.add(Lyr.Conv1D(32, kernel_size=3))
model.add(Lyr.Dense(1))

# Create some dataset and train for it
# https://weiminwang.blog/2017/09/29/multivariate-time-series-forecast-using-seq2seq-in-tensorflow/
# https://github.com/guillaume-chevalier/seq2seq-signal-prediction
