# Intro to ML, Deep Learning, and AI

## What is AI?

AI is the concept of machines carrying out intelligent tasks on their own without explicit programming.

Traditional computer programming is about giving machines specific rules/instructions. AI is about letting machines think.

## What Is Machine Learning (ML)?

ML is when a machine learns from data without using explicit instructions to follow.

## What Is Deep Learning?

@TODO

## What Is Data Mining?

Analyzing data and extracting knowledge/patterns from it. 

Example - analyzing a dataset to understand sales trends.

## Tools for ML?

There's many:

- Matlab Parfor
- GPUs
- MapReduce
- Spark
- GraphLab
- Giraph
- Vowpal
- Parameter Server

Common AI programming tools:

- TensorFlow
- Pytorch
- XLA
- Keras

## What Approaches Are Used in ML?

1. Supervised Learning: when the output variable (the one you want your machine to predict) is labeled in the training data. Example - checking if an email is SPAM, given sample emails that may or may not have the Spam label on them. Techniques: Decision Trees, Random Forests, SVMs, Bayesian Classifiers.
2. Unsupervised Learning: when there is no output variable. Example - grouping customers by their shopping patterns. Techniques: Clustering, Anomaly Detection (think fraud detection), Dimensionality Reduction. 
3. Semi-supervised Learning: when you have a small amount of labeled data and large amount of unlabeled data. Examples - speech recognition and web content classification.
4. Reinforcement Learning: finds a balance between Exploration (unknown territory) and Exploitation (current knowledge). Uses trial and errors, with a defined reward/goal. The idea is to maximize the long-term rewards/goal. Traditional ML is greedy, while RL takes less greedy actions in order to explore all outcomes.

## Classification vs. Regression

1. Classification: Supervised Learning where the output label is discrete or categorical. Example - An email is either spam or it isn't.
2. Regression: continuous or real-valued variables. Example - predicting stock prices, which could be any number at any level of precision (a stock price could be 10.0, 2000.5, or 480.594).

## Online vs. Offline Algorithms

Online = real-time streaming
Offline = batch processing

Online is real-time. If you stream tweets as they are posted to Twitter, and process them with AI, that is an online algo.

If you have a database with 1000 tweets in them, and you feed them all to your AI/ML at the same time, that is an offline algo.

## What Is Bayes Theorem?

@TODO

## What Is Clustering?

@TODO

## What Are Parameters and HyperParameters?

@TODO

## What Does Stochastic Mean?

@TODO
Online. For instance stochastic gradient descent is online.


