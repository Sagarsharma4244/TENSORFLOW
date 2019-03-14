
import os

import tensorflow as tf
from tensorflow import keras
saved_model_path = tf.keras.experimental.export_saved_model(model, "./saved_models")

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)