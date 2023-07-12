#include <iostream>
#include "Eigen/Dense"

#include "NNetwork.h"
#include "Training.h"
#include "Data.h"
#include "Debug.h"


int main()
{
    TrainingData data = loadTrainingDataFromFile("C:\\Users\\Lenovo\\Documents\\dev\\NNetwork2\\TrainingData\\mnist_train.csv");
    normaliseTrainingData(data, DataNormalisationMethod::Z_SCORE);

    // Network setup
    ClassList classes = getClasses();
    size_t inputSz = getInputSz();

    NNetwork network(inputSz, classes);
    network.addLayer(128, 0);
    network.addLayer(64, 0);

    // Data

    // Hyperparameters
    LearningRateList lRList = {0.001, 0.001, 0.001};
    ActFuncList actFuncs = ActFuncList{ActFunc::RELU, ActFunc::RELU, ActFunc::SOFTMAX};
    LossFunc lossFunc = LossFunc::CROSS_ENTROPY;
    InitMethod initMethod = InitMethod::NORMALISED_HE;
    size_t epochs = 1000;
    size_t batchSz = 16;

    // Train
    train(network, data, actFuncs, lossFunc, lRList, initMethod,epochs, batchSz);


}