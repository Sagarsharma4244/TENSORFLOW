# Deep Convolutional Generative Adversarial Network
- Source: https://www.tensorflow.org/alpha/tutorials/generative/dcgan

- DCGAN Paper: https://arxiv.org/pdf/1511.06434.pdf

- Keras Sequental API: https://www.tensorflow.org/guide/keras

- GAN: https://arxiv.org/abs/1406.2661

## What are GANs?
Generative Adversarial Networks (GANs) are one of the most interesting ideas in computer science today. Two models are trained simultaneously by an adversarial process. A generator ("the artist") learns to create images that look real, while a discriminator ("the art critic") learns to tell real images apart from fakes.

During training, the generator progressively becomes better at creating images that look real, while the discriminator becomes better at telling them apart. The process reaches equilibrium when the discriminator can no longer distinguish real images from fakes.


To learn more about GANs, we recommend MIT's Intro to Deep Learning course. : http://introtodeeplearning.com/


## Conclusion

Output for the first epoch will be something like this:

![GAN @001](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/3%20Tensorflow2-Alpha-Tutorials/2%20Generative/3%20DCGAN/image_at_epoch_0001.png "@sagarsharma4244")


According to Tensorflow website, the output at 46th epoch should look something like this:

![GAN @001](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/3%20Tensorflow2-Alpha-Tutorials/2%20Generative/3%20DCGAN/output_46_0.png "@sagarsharma4244")
