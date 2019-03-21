
# Writing layers and models with TensorFlow Keras

Source: https://www.tensorflow.org/alpha/guide/keras/custom_layers_and_models

![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Writing%20Layes%20%26%20Models%20from%20Scratch/1%20Layer%20Class.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Writing%20Layes%20%26%20Models%20from%20Scratch/2%20Non%20trainable%20weights.PNG "@sagarsharma4244")


![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/4%20Tensorflow2-Alpha-Guide/Keras/Writing%20Layes%20%26%20Models%20from%20Scratch/3%20Best%20Practice.PNG "@sagarsharma4244")

## Building Models
### The Model class

class ResNet(tf.keras.Model):

def __init__(self):
    super(ResNet, self).__init__()
    self.block_1 = ResNetBlock()
    self.block_2 = ResNetBlock()
    self.global_pool = layers.GlobalAveragePooling2D()
    self.classifier = Dense(num_classes)

def call(self, inputs):
    x = self.block_1(inputs)
    x = self.block_2(x)
    x = self.global_pool(x)
    return self.classifier(x)
resnet = ResNet()

dataset = ...

resnet.fit(dataset, epochs=10)

resnet.save_weights(filepath)


## Putting it all together: an end-to-end example
Here's what you've learned so far:

- A Layer encapsulate a state (created in init or build) and some computation (in call).
- Layers can be recursively nested to create new, bigger computation blocks.
- Layers can create and track losses (typically regularization losses).
- The outer container, the thing you want to train, is a Model. A Model is just like a Layer, but with added training and serialization utilities.


End-to-end example: we're going to implement a

## Variational AutoEncoder (VAE)
We'll train it on MNIST digits.

