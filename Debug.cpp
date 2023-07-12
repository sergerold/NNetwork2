//
// Created by Lenovo on 08/07/2023.
//

#include "Debug.h"

#include <iostream>

void printNetwork(NNetwork& network)
{
    std::cout << "INPUTS: " << network.getInputs() << std::endl;
    std::cout << "*********************\n";
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        std::cout << "LAYER: " << layerPos << std::endl;
        std::cout << "BIASES: " << network.layer(layerPos).getBiases() << std::endl;
        std::cout << "WEIGHTS: \n" << network.layer(layerPos).getWeights() << std::endl;
        std::cout << "OUTPUTS: " << network.layer(layerPos).getOutputs() << std::endl;
        std::cout << "***************************\n";
    }
}

void printWeightGradients(const WeightGradients& wGrads)
{
    for(size_t layerPos = 0; layerPos < wGrads.numLayers(); ++layerPos)
    {
        std::cout << wGrads.getWeightGradientsForLayer(layerPos) << std::endl << std::endl;
    }
}

void printLayerGradients(const LayerGradients& lGrads)
{
    for(size_t layerPos = 0; layerPos < lGrads.numLayers(); ++layerPos)
    {
        std::cout << lGrads.getLayerGradients(layerPos) << std::endl << std::endl;
    }
}

void printTrainingData(const TrainingData& trData)
{
    for(size_t itemPos = 0; itemPos < trData.size(); ++itemPos)
    {
        std::cout << "Item: " << itemPos << std::endl;
        std::cout << "Inputs (size: " << trData[itemPos].inputs.size() <<"): " << std::endl << trData[itemPos].inputs << std::endl;
        std::cout << "Targets: " << std::endl;
        for(auto labelIt = trData[itemPos].labels.begin(); labelIt != trData[itemPos].labels.end(); ++labelIt)
        {
            std::cout << labelIt->first << " = " << labelIt->second << std::endl;
        }
        std::cout << std::endl;
    }
}