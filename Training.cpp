//
// Created by Lenovo on 08/05/2023.
//

#include "Training.h"

NetNumT mseForTrainingItem(const TrainingItem& trItem, NNetwork& network, std::function<NetNumT(NetNumT)>& actFunc)
{
    network.setInputs(trItem.inputs);
    network.feedforward(actFunc);
    NetNumT mse = 0;
    for(auto targetIt = trItem.targets.begin(); targetIt != trItem.targets.end(); ++targetIt)
    {
        mse += pow ( targetIt->second - network.getOutput(targetIt->first), 2);
    }
    mse /= (NetNumT) trItem.targets.size();
    return mse;
}

NetNumT mseForTrainingData(const TrainingData& trData, NNetwork& network, std::function<NetNumT(NetNumT)>& actFunc)
{
    NetNumT mse = 0;
    for(auto itemIt = trData.begin(); itemIt != trData.end(); ++itemIt)
    {
        mse += mseForTrainingItem(*itemIt, network, actFunc);
    }
    return mse/(NetNumT) trData.size();
}
