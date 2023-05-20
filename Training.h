//
// Created by Lenovo on 08/05/2023.
//

#ifndef NNETWORK2_TRAINING_H
#define NNETWORK2_TRAINING_H

#include "NNetwork.h"

#include <vector>

using TargetValues = std::map<LabelT, NetNumT>;

struct TrainingItem
{
    InputList inputs;
    TargetValues targets;
};

using TrainingData = std::vector<TrainingItem>;

NetNumT mseForTrainingItem(const TrainingItem& trItem, NNetwork& network, std::function<NetNumT(NetNumT)>& actFunc);
NetNumT mseForTrainingData(const TrainingData& trData, NNetwork& network, std::function<NetNumT(NetNumT)>& actFunc);


#endif //NNETWORK2_TRAINING_H
