//
// Created by Lenovo on 09/07/2023.
//

#ifndef NNETWORK2_DATA_H
#define NNETWORK2_DATA_H

#include "NNetwork.h"
#include "Training.h"

enum class DataNormalisationMethod
{
        MINMAX,
        Z_SCORE,
        LOG
};

SingleRowT trainingItemToVector(const std::map<ClassT, NetNumT>& trItem);
bool isTrainingDataValid(const std::map<ClassT, size_t>& networkLabels, const ExampleData& trainingData, size_t networkInputSz);

void normaliseTrainingData(ExampleData& trData, DataNormalisationMethod method);
std::set<std::string> getClasses();
Eigen::Index getInputSz();
ExampleData loadTrainingDataFromFile(const std::string &fName);

bool serialise(std::ofstream& fileOut, NNetwork& network, const ActFuncList& actFuncList);
NNetwork deserialise(std::ifstream& fileIn, ActFuncList& actFuncList);

// DATA PREFIXES

#define PREFIX_ACTFUNCS "ACTFUNCS:"
#define PREFIX_CLASSES "CLASSES:"
#define PREFIX_INPUTSZ "INP_SIZE:"
#define PREFIX_BIASES "LAYER_BIASES"
#define PREFIX_WEIGHTS "LAYER_WEIGHTS:"
#define DELIMITER ','

#endif //NNETWORK2_DATA_H
