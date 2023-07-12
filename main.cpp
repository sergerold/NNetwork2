#include <iostream>
#include "Eigen/Dense"

#include "NNetwork.h"
#include "Training.h"
#include "Data.h"
#include "Debug.h"


int main()
{
    TrainingData data = loadTrainingDataFromFile("C:\\Users\\Lenovo\\Documents\\dev\\NNetwork2\\TrainingData\\iris.csv");

    normaliseTrainingData(data, NormalisationMethod::Z_SCORE);

    // Network setup
    ClassList classes = getClasses();
    size_t inputSz = getInputSz();

    NNetwork network(inputSz, classes);
    network.addLayer(5, 0);

    // Data

    // Hyperparameters
    LearningRateList lRList = {0.001, 0.001};
    ActFuncList actFuncs = ActFuncList{ActFunc::RELU, ActFunc::SOFTMAX};
    LossFunc lossFunc = LossFunc::CROSS_ENTROPY;
    InitMethod initMethod = InitMethod::UNIFORM_XAVIER;
    size_t epochs = 500;
    size_t batchSz = 8;

    // Train
    train(network, data, actFuncs, lossFunc, lRList, initMethod,epochs, batchSz);
    //initialiseWeightsBiases(network,initMethod, actFuncs);
    // printNetwork(network);

}