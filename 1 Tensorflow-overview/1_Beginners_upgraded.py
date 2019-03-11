import tensorflow as tf

print("done Import")
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  									tf.keras.layers.Flatten(input_shape=(28, 28)),
  									tf.keras.layers.Dense(128, activation='relu'),
  									tf.keras.layers.Dropout(0.2),
  									tf.keras.layers.Dense(10, activation='softmax')
])
# from tf.keras.callbacks import TensorBoard
tensorboard = tf.keras.callbacks.TensorBoard(
	log_dir='logs',
	write_grads=True,
	write_graph=True,
	histogram_freq=1,
	write_images=True)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, 
	epochs=5,
	callbacks=[tensorboard])

model.evaluate(x_test, y_test)
