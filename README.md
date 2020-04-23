# robust-ecoc
This is the github repository for the paper "Error Correcting Output Codes Improve Probability Estimation and Adversarial Robustness of Deep Neural Networks"  by Gunjan Verma and Ananthram Swami.

All code is in Python 3.6 using Keras and Tensorflow. Adversarial attacks are done with CleverHans 3.0.1. The two main files to reproduce results in the paper are:
The best way to install Cleverhans is with this command '''pip install git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans'''. This is going to save you a lot of time, because pip install cleverhans has some problems.
You should also note that Cleverhans runs only in Tensorflow 1 for now. I ran my code in Tensorflow 1.15 and keras 2.3.6.
The best way to run this code is by enabling eager execution during the training phase, and disabling it during the attacking phase.

1. TrainModel.py.  Use this script to train a model.  
2. AttackModel.py. Attack (e.g. adversarial PGD attack) the trained model outputted by (1).

In addition, there are a few notable supporting files if one desires to modify the internal implementation

3. Model.py. Abstract base class implementing a baseline or ensemble model. Look at the implementation of "defineModel" in this file to see or modify the neural network architecture used by all ensemble models. 

4. Model_Implementations.py. Implements model-specific methods of (3). Look at the implementation of "defineModelBaseline" in this file to see or modify the neural network architecture used by all baseline models. 
