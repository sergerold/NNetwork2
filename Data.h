//
// Created by Lenovo on 09/07/2023.
//

#ifndef NNETWORK2_DATA_H
#define NNETWORK2_DATA_H

#include "NNetwork.h"
#include "Training.h"

enum class NormalisationMethod
{
        MINMAX,
        Z_SCORE,
        LOG
};

SingleRowT trainingItemToVector(const std::map<ClassT, NetNumT>& trItem);
bool isTrainingDataValid(const std::map<ClassT, size_t>& networkLabels, const TrainingData& trainingData, size_t networkInputSz);

void normaliseTrainingData(TrainingData& trData, NormalisationMethod method);
std::set<std::string> getClasses();
NetNumT getInputSz();
TrainingData loadTrainingDataFromFile(std::string fName);

#endif //NNETWORK2_DATA_H
