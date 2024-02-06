//
// Created by Lenovo on 08/07/2023.
//

#ifndef NNETWORK2_DEBUG_H
#define NNETWORK2_DEBUG_H

#include "NNetwork.h"
#include "Training.h"

void printNetwork(NNetwork& network);
void printWeightGradients(const NetworkWeightGradients& wGrads);
void printLayerGradients(const NetworkLayerGradients& lGrads);

void printOutputs(const NNetwork& network);

void printTrainingData(const ExampleData& trData);

#endif //NNETWORK2_DEBUG_H
