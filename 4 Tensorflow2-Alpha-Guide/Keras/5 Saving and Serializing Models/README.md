# Saving and Serializing Models with TensorFlow Keras
Source: https://www.tensorflow.org/alpha/guide/keras/saving_and_serializing#weights-only_saving


Part 1 : Saving and Serialization for Sequential Models and Functional API.

Part 2 : Saving for Custom subclass of models. The API is slightly different to Sequential or Functional.


# Part I: Saving Sequential models or Functional models

Let's consider the following model:
```
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()

```
Train the Model.

```
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1,
                   verbose = 2)
```

## Whole-model saving

This file includes:

- The model's architecture
- The model's weight values (which were learned during training)
- The model's training config (what you passed to compile), if any
- The optimizer and its state, if any (this enables you to restart training where you left off)


```
# Save the model
model.save('path_to_my_model.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')

```

## Export to SavedModel

## Architecture-only saving

## Weights-only saving

## Weights-only saving in SavedModel format

# Saving Subclassed Models

Open the notebook for details
