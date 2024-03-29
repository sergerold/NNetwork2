#include <fstream>
#include <iostream>
#include <omp.h>

#include "NNetwork.h"
#include "Training.h"
#include "Data.h"



int main()
{
    omp_set_num_threads(4);
    Eigen::setNbThreads(4);

    // Data
    std::cout << "Loading and normalising data...\n \n";

    ExampleData trainingData = loadTrainingDataFromFile("../TrainingData/mnist_train_3.csv");
    normaliseTrainingData(trainingData, DataNormalisationMethod::Z_SCORE);

    ExampleData testData = loadTrainingDataFromFile("../TrainingData/mnist_test.csv");
    normaliseTrainingData(testData, DataNormalisationMethod::Z_SCORE);

    // Network setup
    ClassList classes = getClasses();
    size_t inputSz = getInputSz();

    NNetwork network(inputSz, classes);
    network.addLayer(128, 0);
    network.addLayer(64, 1);

    network.summarise(std::cout);
    std::cout << "\n";

    // Hyperparameters
    LearningRateList lRList = {0.01, 0.01, 0.01};
    ActFuncList actFuncs = ActFuncList{ ActFunc::RELU, ActFunc::RELU,  ActFunc::SOFTMAX };
    LossFunc lossFunc = LossFunc::CROSS_ENTROPY;
    InitMethod initMethod = InitMethod::UNIFORM_HE;
    size_t epochs = 10;
    size_t batchSz = 16;
    NetNumT momentum = 0;
    NetNumT dropOutRate = 0;

    // Train
    train(network, trainingData, actFuncs, lossFunc, lRList, momentum, initMethod, epochs, batchSz, testData, dropOutRate);

    //std::ofstream fOut ("../model.dat");
    //serialise(fOut, network, actFuncs);
}