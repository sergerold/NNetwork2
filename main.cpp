#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "Eigen/Dense"

#include "NNetwork.h"
#include "Training.h"
#include "Data.h"
#include "Debug.h"

int main()
{
    // Data

    ExampleData trainingData = loadTrainingDataFromFile("C:\\Users\\Lenovo\\Documents\\dev\\NNetwork2\\TrainingData\\mnist_train_3.csv");
    normaliseTrainingData(trainingData, DataNormalisationMethod::Z_SCORE);

    ExampleData testData = loadTrainingDataFromFile("C:\\Users\\Lenovo\\Documents\\dev\\NNetwork2\\TrainingData\\mnist_test.csv");
    normaliseTrainingData(testData, DataNormalisationMethod::Z_SCORE);


    // Network setup
    ClassList classes = getClasses();
    size_t inputSz = getInputSz();

    NNetwork network(inputSz, classes);
    network.addLayer(128, 0);
    network.addLayer(64, 1);

    // Hyperparameters
    LearningRateList lRList = {0.01, 0.01, 0.01};
    ActFuncList actFuncs = ActFuncList{ActFunc::RELU, ActFunc::RELU,  ActFunc::SOFTMAX };
    LossFunc lossFunc = LossFunc::CROSS_ENTROPY;
    InitMethod initMethod = InitMethod::UNIFORM_HE;
    size_t epochs = 30;
    size_t batchSz = 8;

    network.summarise(std::cout);
    // Train
    //train(network, trainingData, actFuncs, lossFunc, lRList, initMethod, epochs, batchSz, testData);
    std::ifstream in("model 2.net");
    ActFuncList actFuncList2;
    NNetwork net2 = deserialise(in, actFuncList2);

    std::cout << "   --> Accuracy: " << std::fixed << calculateAccuracyForExampleData(net2, trainingData, actFuncs) << "%" << std::endl;
    std::cout << "   --> Accuracy: " << std::fixed << calculateAccuracyForExampleData(net2, testData, actFuncs) << "%" << std::endl;
}