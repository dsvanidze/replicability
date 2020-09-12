import numpy as np
import tensorflow as tf
from datetime import datetime

begin = datetime.now()
a = tf.constant(np.random.rand(10000, 10000))
b = tf.constant(np.random.rand(10000, 10000))
print("Constant creation took:", datetime.now() - begin)

begin1 = datetime.now()
tf.matmul(a, b)
print("Multiplication took:", datetime.now() - begin1)

print("Overall took:", datetime.now() - begin)
