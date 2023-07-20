//
// Created by Lenovo on 08/07/2023.
//

#ifndef NNETWORK2_DEBUG_H
#define NNETWORK2_DEBUG_H

#include "NNetwork.h"
#include "Training.h"

void printNetwork(NNetwork& network);
void printWeightGradients(const WeightGradients& wGrads);
void printLayerGradients(const LayerGradients& lGrads);

void printOutputs(const NNetwork& network);

void printTrainingData(const TrainingData& trData);
void printTrainingItem(const TrainingItem trainingItem);

#endif //NNETWORK2_DEBUG_H
