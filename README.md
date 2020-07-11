# Local Competition and Uncertainty for Adversarial Robustness

This repository is the official implementation of Local Competition and Uncertainty for Adversarial Robustness.


## Requirements

We have provided the conda environment `adver_environment.yml`, in which we conducted the experiments. You can also see the dependencies in the file `requirements.txt`. There is a possibility that the command ```pip install -r requirements.txt``` may fail. For installing the conda environment use ```conda env create -f adver_environment.yml```.

If you want to install CleverHans separately, please follow the [instructions](https://github.com/tensorflow/cleverhans) and you should probably use 


```
pip install git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans
```

## Files

We give a brief description of the important files provided in this repository:

- Model.py : Abstract base class implementing a baseline or ensemble model. Look at the implementation of "defineModel" in this file to see or modify the neural network architecture used by all ensemble models. 

- Model_Implementations.py : Implements model-specific methods of Model.py. Look at the implementation of "defineModelBaseline" in this file to see or modify the neural network architecture used by all baseline models.

- Attack_Model.ipynb/Attack_Model.py : Code that runs the attacks. We have used mostly the notebook file.

- Train_Model.ipynb/Train_Model.py : Code for training the models. We have used mostly the notebook file.

- automatic_plot.ipynb : It is the code for the probability distribution figures from the text.

- distributions.py : File that contains the functions for probabilities, distributions, sampling, etc.

- lwta_conv2d_activation.py : main code for the LWTA activation for convolutional layers.

- lwta_dense_activation.py: main code for the LWTA activation for dense layers.

- sbp_lwta_con2d_layer.py : file that contains the code our Convolution layer with IBP and LWTA.

- sbp_lwta_dense_layer.py : file that contains the code our Dense layer with IBP and LWTA.

## Training

To train the model(s) in the paper, run either the `Train_Model.ipynb` or `Train_Model.py`. For the latter run:

```train
python Train_Model.py
```
It is important to mention that the code runs eagerly. So for training comment the line 9 in `Model_Implementations.py`, and line 11 in `Model.py`. For MNIST dataset uncomment the lines for MNIST dataset parameters in file `Train_Model.py`, or `Train_Model.ipynb`(whichever you use), and comment the lines for CIFAR10 dataset parameters. For CIFAR10 do the opposite. For Mnist also you need to uncomment line 49 in `Model_Implementations.py` and comment line 47; also uncomment lines 114,115 in `Model.py` and comment lines 111,112. For CIFAR10 do the opposite of the previous sentence. There also some helpful comments on the code to guide through this process.

If you want to run a pretrained model, make sure to uncomment line 221 in `Model.py`, or else the model will start training from the beginning.



## Adversarial Attacks.

To run the attacks you can use either `Attack_Model.ipynb` or `Attack_Model.py`. We used mostly the notebook. In order to run the attacks you have to disable eager execution from the files `Model_Implementations.py` and `Model.py`. There are some information/instructions inside the attacks file. It works just like the Training file. Only in this case you have to take the model and its parameters from the `Train_Model.py`, or `Train_Model.ipynb`and put them inside the attack file. After the execution of the attacks you can see the cells that run the plot the figures and the probabilities from the LWTA activations as we have presented them in the paper.

## Pre-trained Models

Having space constraints on the supplementary material(100 mb) we have not included all the pretrained models. We uploaded only a few models with 4 competing units since they do not produce good results, and all the models with 2 competing units.


## References

We have used code from [here](https://github.com/Gunjan108/robust-ecoc) and [here](https://github.com/konpanousis/SB-LWTA)