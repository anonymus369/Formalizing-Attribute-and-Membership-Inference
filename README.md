# Formalizing Attribute and Membership Inference

This repository contains the code necessary to reproduce the results from "Formalizing attribute and Membership Inference Attacks on Machine Learning Models".

The notebooks will reproduce the experimental results presented in section 4 and appendix B from the paper:

* Gaussian.ipynb corresponds to section 4.1, Memebership inference attacks against a Linear Regression model on Gaussian data. The success rate of the optimal attacker and its corresponding lower bound are computed.
* Cifar10.ipynb corresponds to section 4.2, Membership inference attacks agains DNNs for image classification (CIFAR10). The likelihood attack strategy is implemented, and the lower bound to the success rate of the optimal attacker is computed.
* PenDigits.ipynb corresponds to section 4.3, Attribute and Membership inference attacks against a NN for classification of PenDigits. Four different attribute inference strategies are implemented. In addition, the likelihood attack strategy is implemented, and the lower bound to the success rate of the optimal attacker is computed.
* MNIST.ipynb and MNISTFashion.ipynb correspond appendix B: Additional experiments on MNIST and FashionMNIST. See the reference for Cifar10.ipynb.
