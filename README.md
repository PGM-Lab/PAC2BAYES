# PAC2BAYES
This repository contains the Python code to reproduce all the figures and experiments presented 
in the paper: 

Masegosa, Andrés. R., Learning under Model Misspecification: Applications to Variational 
and Ensemble methods. https://arxiv.org/abs/1912.08335


## Dependencies

The code is written in Python 3 and uses the following libraries:
 * [Tensorflow v1.15.0](https://www.tensorflow.org/)
 * [Tensorflow-Probability v0.8.0](https://www.tensorflow.org/probability)
 * [Numpy v1.18.4](https://numpy.org/)   

## Directory structure

This repository has the following directory structure
 * *README*: This file.
 * *scripts*: Folder containing the python scripts to reproduce the results of the empirical evaluation of the paper. More details below. 
 * *results*: Foder containing the ouput of the python scripts included in the folder 'scritps'. 
 * *notebooks*: Folder containing Jupyter notebooks where all the figures and analysis with artificial data sets can be reprodcued. More details below. 
   


## Reproduce experiments with real data sets

Execute the following python scrpits, which are grouped by algorithm and by task. Running each script you get back results for each data set and for each method used in the paper.   

*  **PAC^2-Varitional** and **PAC^2_T-Variational** learning algorithms:

    - Supervised classification task: [PAC2-Variational-Supervised.py](https://github.com/PGM-Lab/PAC2BAYES/blob/master/scripts/PAC2-Variational-Supervised.py).
        ```console
        > python ./scripts/PAC2-Variational-Supervised.py
        ```

    - Self-Supervised classification task with Normal likelihood: [PAC2-Variational-SelfSupervisedNormal.py](https://github.com/PGM-Lab/PAC2BAYES/blob/master/scripts/PAC2-Variational-SelfSupervisedNormal.py).
        ```console
        > python ./scripts/PAC2-Variational-SelfSupervisedNormal.py
        ```

    - Self-Supervised classification task with Binomial likelihood: [PAC2-Variational-SelfSupervisedBinomial.py](https://github.com/PGM-Lab/PAC2BAYES/blob/master/scripts/PAC2-Variational-SelfSupervisedBinomial.py).
        ```console
        > python ./scripts/PAC2-Variational-SelfSupervisedBinomial.py
        ```

*  **PAC^2-Ensemble** and **PAC^2_T-Ensemble** learning algorithms:

    - Supervised classification task: [PAC2-Ensemble-Supervised.py](https://github.com/PGM-Lab/PAC2BAYES/blob/master/scripts/PAC2-Ensemble-Supervised.py).
        ```console
        > python ./scripts/PAC2-Ensemble-Supervised.py
        ```
       
    - Self-Supervised classification task with Normal likelihood: [PAC2-Ensemble-SelfSupervisedNormal.py](https://github.com/PGM-Lab/PAC2BAYES/blob/master/scripts/PAC2-Ensemble-SelfSupervisedNormal.py).
        ```console
        > python ./scripts/PAC2-Ensemble-SelfSupervisedNormal.py
        ```

    - Self-Supervised classification task with Binomial likelihood: [PAC2-Ensemble-SelfSupervisedBinomial.py](https://github.com/PGM-Lab/PAC2BAYES/blob/master/scripts/PAC2-Ensemble-SelfSupervisedBinomial.py).
        ```console
        > python ./scripts/PAC2-Ensemble-SelfSupervisedBinomial.py
        ```

## Notebooks 

Each of the figures with artificial data illustrating the algorithms can be reproduced using the following notebooks:
 
 * [First-order vs Second-order bounds](https://github.com/PGM-Lab/PAC2BAYES/blob/master/notebooks/FirstOrdervsSeconOrderBounds.ipynb) [[Open in Google Colab](http://colab.research.google.com/github/PGM-Lab/PAC2BAYES/blob/master/notebooks/FirstOrdervsSeconOrderBounds.ipynb)].
 
 * [Bayesian Linear Regression](https://github.com/PGM-Lab/PAC2BAYES/blob/master/notebooks/Bayesian-LinearRegression.ipynb) [[Open in Google Colab](http://colab.research.google.com/github/PGM-Lab/PAC2BAYES/blob/master/notebooks/Bayesian-LinearRegression.ipynb)].
 
 * [PAC2-Variational Linear Regression](https://github.com/PGM-Lab/PAC2BAYES/blob/master/notebooks/PAC2-Variational-LinearRegression.ipynb) [[Open in Google Colab](http://colab.research.google.com/github/PGM-Lab/PAC2BAYES/blob/master/notebooks/PAC2-Variational-LinearRegression.ipynb)].
 
 * [PAC2-Variational - Sinusoidal Data - Neural Network](https://github.com/PGM-Lab/PAC2BAYES/blob/master/notebooks/PAC2-Variational-SinusoidalData-NeuralNetwork.ipynb) [[Open in Google Colab](http://colab.research.google.com/github/PGM-Lab/PAC2BAYES/blob/master/notebooks/PAC2-Variational-SinusoidalData-NeuralNetwork.ipynb)].

 * [PAC2-Ensemble - Sinusoidal Data - Neural Network](https://github.com/PGM-Lab/PAC2BAYES/blob/master/notebooks/PAC2-Ensemble-SinusoidalData-NeuralNetwork.ipynb) [[Open in Google Colab](http://colab.research.google.com/github/PGM-Lab/PAC2BAYES/blob/master/notebooks/PAC2-Ensemble-SinusoidalData-NeuralNetwork.ipynb)].

 * [PAC2-Ensemble - MultiModal Data - Neural Network](https://github.com/PGM-Lab/PAC2BAYES/blob/master/notebooks/PAC2-Ensemble-MultiModalData-NeuralNetwork.ipynb) [[Open in Google Colab](http://colab.research.google.com/github/PGM-Lab/PAC2BAYES/blob/master/notebooks/PAC2-Ensemble-MultiModalData-NeuralNetwork.ipynb)].

 
 

 
 
 
 
