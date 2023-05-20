//
// Created by Lenovo on 10/04/2023.
//

#include "NLayer.h"

#include <iostream>

NLayer::NLayer(size_t layerSz, size_t numIncomingWeightsToEachNeuron)
{
    mLayerBiases.resize(1, layerSz);
    mLayerOutputs.resize(1, layerSz);
    mLayerWeights.resize(numIncomingWeightsToEachNeuron, layerSz);
}

SingleRowT& NLayer::getBiases()
{
    return mLayerBiases;
}

const SingleRowT& NLayer::getOutputs()
{
    return mLayerOutputs;
}

Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic>& NLayer::getWeights()
{
    return mLayerWeights;
}

Eigen::Ref<Eigen::VectorXd> NLayer::getWeightsForNeuron(size_t neuronPos)
{
    return mLayerWeights.col(neuronPos);
}

size_t NLayer::size()
{
    return mLayerOutputs.size();
}

void NLayer::resizeLayer(size_t newLayerSz)
{
    mLayerBiases.resize(1, newLayerSz);
    mLayerOutputs.resize(1, newLayerSz);
    mLayerWeights.resize(mLayerWeights.rows(), newLayerSz);
}

void NLayer::resizeNumWeightsPerNeuron(size_t newWeightsSz)
{
    mLayerWeights.resize(newWeightsSz, mLayerWeights.cols());
}