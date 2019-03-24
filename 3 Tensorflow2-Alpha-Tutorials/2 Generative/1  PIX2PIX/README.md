# Pix2Pix
Source: https://www.tensorflow.org/alpha/tutorials/generative/pix2pix

This notebook demonstrates image to image translation using conditional GAN's, as described in Image-to-Image Translation with Conditional Adversarial Networks. Using this technique we can colorize black and white photos, convert google maps to google earth, etc. Here, we convert building facades to real buildings.

Implementation of paper:
## Image-to-Image Translation with Conditional Adversarial Networks
Source: https://arxiv.org/abs/1611.07004 , also in this repo

Each epoch takes around 58 seconds on a single P100 GPU.

Below is the output generated after training the model for 200 epochs.

I was unable to download the data with through jupyter notebook so...
## Download data Manually
You can just download the data manually by saving this link and save it and extract it in the folder with current python file.
https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz

and just change the directoy according to that. 
` PATH = r"C:\Users\HP\Desktop\Code\TENSORFLOW\3 Tensorflow2-Alpha-Tutorials\2 Generative\1  PIX2PIX\facades"`

### Dataset
![PIX2PIX](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/3%20Tensorflow2-Alpha-Tutorials/2%20Generative/1%20%20PIX2PIX/pix2pix_1.png "@sagarsharma4244")


![PIX2PIX](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/3%20Tensorflow2-Alpha-Tutorials/2%20Generative/1%20%20PIX2PIX/pix2pix_2.png "@sagarsharma4244")


### Training

![Discriminator](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/3%20Tensorflow2-Alpha-Tutorials/2%20Generative/1%20%20PIX2PIX/Discriminator.png "@sagarsharma4244")


![Generator](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/3%20Tensorflow2-Alpha-Tutorials/2%20Generative/1%20%20PIX2PIX/Generator.png "@sagarsharma4244")
