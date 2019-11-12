# Feature Selection and Feature Extraction

## Introduction
This repository contains different feature selection and extraction methods.  
We using MNIST dataset for training and testing.
A simple classifier, Naive Bayes is used for experiments in order to magnify the effectiveness of the feature selection and extraction methods.  
Methods using in this responsitory are based on the paper: Feature Selection and Feature Extraction in Pattern Analysis: A Literature Review.

## Libraries
`sklearn`  
`numpy`  
`scipy`  
`mlxtend`  
`pyswarms`  
`metric_learn`  
`keras`  
`tensorflow`

## Notebooks
`feature_selection.ipynb`: Introducing feature selection methods containing Correlation Criteria, Mutual Information, Chi-square Statistics, Fast Correlation-based Filter, Sequential Forward Selection, Particles Swarm Optimization, Genetic Algorithms.  

`feature_extraction.ipynb`: Introducing feature selection methods containing Principal Components Analysis, Kernel Principal Components Analysis, Multidimensional Scaling, Isomap, Locally Linear Embedding, Laplacian Eigenmap, t-distributed Stochastic Neighbor Embedding, Fisher Linear Discriminant Analysis, Supervised Principal Component Analysis, Metric learning.

`autoencoder_tf2.ipynb`: Introducing autoencoder used for feature extraction.

`rbm_dimensional_reduction.ipynb`: Introducing Restricted Boltzmann Machine used for dimensional reduction and reconstruction.

`autoencoder_rbm.ipynb`: Introducing autoencoder with weights initialized by pretrained Restricted Boltzmann Machine.

`other_files`: Supporting classes and methods.

## Reference
1. B. Ghojogh, M. N. Samad, S. A. Mashhadi, T. Kapoor, W. Ali, F. Karray, M. Crowley. Feature Selection and Feature Extraction in Pattern Analysis: A Literature Review.
2. Fast Correlation-based Filter algorithm. Paper: Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution. Yu & Liu (ICML 2003). Github: https://github.com/shiralkarprashant/FCBF
3. Supervised Principal Components Analysis. Paper: Supervised Principal Component Analysis: Visualization, Classification and Regression on Subspaces and Submanifolds. Github: https://github.com/kumarnikhil936/Supervised-PCA-Python
4. Genetic Algorithm Implementation in Python. Link: https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
5. Implementing the Particle Swarm Optimization (PSO) Algorithm in Python. Link: https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6
6. G. E. Hinton and R. R. Salakhutdinov, “Reducing the dimensionality of data with neural networks” Science, vol. 313, no. 5786, pp. 504–507, 2006.
7. Tensorflow Example: https://github.com/aymericdamien/TensorFlow-Examples