
import os

import tensorflow as tf
from tensorflow import keras
saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)