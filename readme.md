# NIHRIO at SemEval-2018 Task 3: A Simple and Accurate Neural Network Model for Irony Detection in Twitter

We propose to use the Multilayer Perceptron (MLP) model to handle both the ironic tweet detection task. The figure bellow presents an overview of our model architecture including an input layer, two hidden layers and a softmax output layer.

<p align="center">
<img src="https://github.com/NIHRIO/IronyDetectionInTwitter/blob/master/description/mlp.png" alt="Overview of our model architecture for irony detection in tweets" width="80%"/>
</p>

Given a tweet, the input layer represents the tweet by a feature vector which concatenates <b>lexical</b>, <b>syntactic</b>, <b>semantic</b> and <b>polarity</b> feature representations.

The two hidden layers with ReLU activation function take the input feature vector to select the most important features which are then fed into the softmax layer for ironic detection and classification.



