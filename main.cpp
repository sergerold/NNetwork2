#include <iostream>
#include <fstream>
#include <iomanip>
#include "Eigen/Dense"

#include "NNetwork.h"
#include "Training.h"
#include "Data.h"
#include "Debug.h"


int main()
{
    Eigen::initParallel();
    Eigen::setNbThreads(4);

    TrainingData data = loadTrainingDataFromFile("C:\\Users\\Lenovo\\Documents\\dev\\NNetwork2\\TrainingData\\mnist_train_2.csv");
    normaliseTrainingData(data, DataNormalisationMethod::Z_SCORE);

    // Network setup
    ClassList classes = getClasses();
    size_t inputSz = getInputSz();

    NNetwork network(inputSz, classes);
    network.addLayer(128, 0);
    network.addLayer(64, 1);

    // Hyperparameters
    LearningRateList lRList = {0.001, 0.001, 0.001};
    ActFuncList actFuncs = ActFuncList{ActFunc::RELU, ActFunc::RELU, ActFunc::SOFTMAX};
    LossFunc lossFunc = LossFunc::CROSS_ENTROPY;
    InitMethod initMethod = InitMethod::UNIFORM_HE;
    size_t epochs = 100;
    size_t batchSz = 8;

    // Train
    train(network, data, actFuncs, lossFunc, lRList, initMethod,epochs, batchSz);
//    std::ifstream in("mnist_error.net");
//    ActFuncList funcList2;
//    NNetwork network2{deserialise(in, funcList2)};
//
//    size_t dataPos = 3020;
//    network2.setInputs(data[dataPos].inputs);
//    network2.feedforward(actFuncs);
//    std::cout << calculateLossForTrainingItem(data[dataPos].labels, lossFunc, network2.outputLayer().getOutputs());

}