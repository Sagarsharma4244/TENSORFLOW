# Training and Evaluation with TensorFlow Keras
Source: https://www.tensorflow.org/alpha/guide/keras/training_and_evaluation

![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/1%20End%20to%20end%20Example.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/2%20Custom%20loss%20and%20metrics.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/2_2%20Custom%20loss%20and%20metrics.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/3%20FunctionalAPI.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/4%20Setting%20apart%20Validation.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/5%20Training%20%26%20evaluation%20from%20Datasets.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/6%20Specific%20number%20of%20batches.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/7%20Sample%20Weight%20sample%20classes.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Training%20and%20Evaluation%20with%20TensorFlow%20Keras/7_2%20Sample%20Weight%20sample%20classes.PNG "@sagarsharma4244")


## Part II: Writing your own training & evaluation loops from Scratch
### Using the GradientTape: a first end-to-end example
```
# Get the model.
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Prepare the training dataset.
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Iterate over epochs.
for epoch in range(3):
  print('Start of epoch %d' % (epoch,))
  
  # Iterate over the batches of the dataset.
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    with tf.GradientTape() as tape:

      # Run the forward pass of the layer.
      # The operations that the layer applies
      # to its inputs are going to be recorded
      # on the GradientTape.
      logits = model(x_batch_train)  # Logits for this minibatch

      # Compute the loss value for this minibatch.
      loss_value = loss_fn(y_batch_train, logits)

      # Use the gradient tape to automatically retrieve
      # the gradients of the trainable weights with respect to the loss.
      grads = tape.gradient(loss_value, model.trainable_variables)

      # Run one step of gradient descent by updating
      # the value of the weights to minimize the loss.
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Log every 200 batches.
    if step % 200 == 0:
        print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
        print('Seen so far: %s samples' % ((step + 1) * 64))
     
