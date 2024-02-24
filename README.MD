# Feedforward Neural Network

## Introduction and functionality

This is a C++ feedforward neural library. This library uses the [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page) to increase performance.

This library supports the following functionality:
- An arbitrary number of hidden layers
- A number of common weight initialisation methods (He, Xavier)
- Momentum gradients
- A dropout rate
- Mini-batch

The following activation functions are supported:
- Sigmoid
- ReLu
- Softmax 

The following loss functions are supported:
- MSE
- Cross entropy loss

Basic data functionality including:
- Loading data from a CSV file
- Normalisation methods (Log, MinMax, Z_SCORE)

## Quick start guide

Creating a neural network:

```
    ClassList classes = getClasses(); // gets a list of classes from DataSpecs.h
    size_t inputSz = getInputSz(); // gets the number of inputs from DataSpecs.h

    NNetwork network(inputSz, classes); // create the network object
    network.addLayer(128, 0); // hidden layer 0 with 128 neurons
    network.addLayer(64, 1); // hidden layer 1 with 65 neurons

    network.summarise(std::cout); // summarise the network
```

Loading and normalising data

```
    ExampleData trainingData = loadTrainingDataFromFile("../TrainingData/mnist_train_3.csv"); // load training data
    normaliseTrainingData(trainingData, DataNormalisationMethod::Z_SCORE); // normalise training data using Z_SCORE

    ExampleData testData = loadTrainingDataFromFile("../TrainingData/mnist_test.csv"); // load testing data
    normaliseTrainingData(testData, DataNormalisationMethod::Z_SCORE);
```

Hyperparameters:

```
    // Hyperparameters
    LearningRateList lRList = {0.01, 0.01, 0.01}; // each hidden layer and output layer can have a learning rate
    ActFuncList actFuncs = ActFuncList{ ActFunc::RELU, ActFunc::RELU,  ActFunc::SOFTMAX }; // each layer can have its own activation function
    LossFunc lossFunc = LossFunc::CROSS_ENTROPY; // set the loss function
    InitMethod initMethod = InitMethod::UNIFORM_HE; // set the weight initiatialisation method
    size_t epochs = 10; // number of epochs
    size_t batchSz = 16; // mini batch size
    NetNumT momentum = 0; // momentum
    NetNumT dropOutRate = 0; // drapout rate
```

Training and saving the network:

```
    train(network, trainingData, actFuncs, lossFunc, lRList, momentum, initMethod, epochs, batchSz, testData, dropOutRate);

    std::ofstream fOut ("../model.dat");
    serialise(fOut, network, actFuncs); // save the network
```


