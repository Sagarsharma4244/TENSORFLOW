
import os

import tensorflow as tf
from tensorflow import keras
# this is updated in the new tensorflow2.0 
# Source : https://www.tensorflow.org/alpha/guide/upgrade
# Command: tf_upgrade_v2 --infile new.py --outfile upgraded.py
saved_model_path = tf.keras.experimental.export_saved_model(model, "./saved_models")