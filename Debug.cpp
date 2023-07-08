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