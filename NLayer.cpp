//
// Created by Lenovo on 10/04/2023.
//

#include "NLayer.h"

#include <iostream>

NLayer::NLayer(size_t layerSz, size_t numIncomingWeightsToEachNeuron)
{
    mLayerBiases.resize(1, Eigen::Index (layerSz) );
    mLayerOutputs.resize(1, Eigen::Index (layerSz) );
    mLayerWeights.resize(Eigen::Index (numIncomingWeightsToEachNeuron), Eigen::Index (layerSz) );
}

const SingleRowT& NLayer::getBiases()
{
    return mLayerBiases;
}

void NLayer::setBiases(const SingleRowT& biases)
{
    if(biases.size() != mLayerBiases.size())
    {
        throw std::logic_error("Biases different size to that specified in Layer");
    }
    mLayerBiases = biases;
}


const SingleRowT& NLayer::getOutputs() const
{
    return mLayerOutputs;
}

const LayerWeightsT& NLayer::getWeights()
{
    return mLayerWeights;
}

void NLayer::setWeights(const LayerWeightsT& weights)
{
    if(weights.rows() != mLayerWeights.rows() || weights.cols() != mLayerWeights.cols())
    {
        throw std::logic_error("Weights different size to that specified in Layer");
    }
    mLayerWeights = weights;
}

Eigen::Ref<Eigen::VectorXd> NLayer::getWeightsForNeuron(size_t neuronPos)
{
    return mLayerWeights.col(Eigen::Index (neuronPos) );
}

size_t NLayer::size()
{
    return mLayerOutputs.size();
}

void NLayer::resizeLayer(size_t newLayerSz)
{
    mLayerBiases.resize(1, Eigen::Index(newLayerSz) );
    mLayerOutputs.resize(1, Eigen::Index(newLayerSz) );
    mLayerWeights.resize(mLayerWeights.rows(), Eigen::Index(newLayerSz) );
}

void NLayer::resizeNumWeightsPerNeuron(size_t newWeightsSz)
{
    mLayerWeights.resize(Eigen::Index (newWeightsSz), mLayerWeights.cols());
}


