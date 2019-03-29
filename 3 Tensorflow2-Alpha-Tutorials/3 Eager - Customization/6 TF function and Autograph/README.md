
# tf.function
Source: https://www.tensorflow.org/alpha/tutorials/eager/tf_function

```
!pip install -q tensorflow==2.0.0-alpha0
import tensorflow as tf
```
## Polymorphism
tf.function tries to be as generic as a Python function. You can call Python functions with all sorts of signatures, and Python will usually do something reasonable. tf.function does this type of polymorphism for you even though the underlying TensorFlow graphs it generates are specific to the particular types in its signature.

You can call a function with arguments of different types to see what is happening.


Functions are polymorphic
```
@tf.function
def add(a):
  return a + a

print("add 1", add(1))
print("add 1.1", add(1.1))
print("add string tensor", add(tf.constant("a")))
c = add.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
c(a=tf.constant("a"))  # aa
```

Functions can be faster than eager code, for graphs with many small ops
```

import timeit
conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function
def conv_fn(image):
  return conv_layer(image)

image = tf.zeros([1, 200, 200, 100])
# warm up
conv_layer(image); conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")

lstm_cell = tf.keras.layers.LSTMCell(10)

@tf.function
def lstm_fn(input, state):
  return lstm_cell(input, state)

input = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2
# warm up
lstm_cell(input, state); lstm_fn(input, state)
print("eager lstm:", timeit.timeit(lambda: lstm_cell(input, state), number=10))
print("function lstm:", timeit.timeit(lambda: lstm_fn(input, state), number=10))
```
## Control flow and autograph
```
# Simple loop

@tf.function
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x

f(tf.random.uniform([10]))
```
