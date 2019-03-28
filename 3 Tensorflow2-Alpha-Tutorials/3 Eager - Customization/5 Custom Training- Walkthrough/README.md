# Custom training: walkthrough
Source: https://www.tensorflow.org/alpha/tutorials/eager/custom_training_walkthrough#tensorflow_programming


## TensorFlow programming
This guide uses these high-level TensorFlow concepts:

- Use TensorFlow's default eager execution development environment,
- Import data with the Datasets API,
- Build models and layers with TensorFlow's Keras API.

This tutorial is structured like many TensorFlow programs:

- Import and parse the data sets.
- Select the type of model.
- Train the model.
- Evaluate the model's effectiveness.
- Use the trained model to make predictions.

## The Iris classification problem
- Iris setosa
- Iris virginica
- Iris versicolor

![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/3%20Tensorflow2-Alpha-Tutorials/3%20Eager%20-%20Customization/5%20Custom%20Training-%20Walkthrough/iris_three_species.jpg "@sagarsharma4244")


You can start to see some clusters by plotting a few features from the batch:
```
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length");
```
![Tensorflow LOGO](https://github.com/Sagarsharma4244/TENSORFLOW/blob/master/3%20Tensorflow2-Alpha-Tutorials/3%20Eager%20-%20Customization/5%20Custom%20Training-%20Walkthrough/few%20features%20from%20the%20batch.png "@sagarsharma4244")
