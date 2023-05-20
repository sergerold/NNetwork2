//
// Created by Lenovo on 10/04/2023.
//

#include <iostream>

#include "NNetwork.h"
#include "NLayer.h"

NNetwork::NNetwork(const SingleRowT& inputs, const LabelList& labels)
{
    // add input layer
    NLayer inputLayer(inputs.cols(), 0);
    inputLayer.mLayerOutputs = inputs;
    mNLayer.push_back(inputLayer);

    // add output layer
    NLayer outputLayer(labels.size(), inputs.cols());
    mNLayer.push_back(outputLayer);

    // add output labels
    for(size_t lPos = 0; lPos < labels.size(); ++ lPos)
    {
        mOutputLabels.emplace(*std::next(labels.begin(), lPos), lPos);
    }
}

NLayer& NNetwork::layer(size_t layer)
{
    if (layer >= mNLayer.size() - 1)
    {
        throw std::out_of_range("No such layer");
    }
    return mNLayer[layer + INPUT_LAYER_OFFSET];
}

const SingleRowT& NNetwork::getInputs()
{
    return mNLayer[0].getOutputs();
}

void NNetwork::setInputs(const SingleRowT& inputs)
{
    if (inputs.cols() != getInputs().cols())
    {
        throw std::out_of_range("Num inputs does not match current input layer size");
    }
    mNLayer[0].mLayerOutputs = inputs;
}

size_t NNetwork::numLayers()
{
    return mNLayer.size() - INPUT_LAYER_OFFSET;
}

bool NNetwork::addLayer(size_t layerSz, size_t insertLayerBefore)
{
    if (insertLayerBefore + INPUT_LAYER_OFFSET >= mNLayer.size())
    {
        return false;
    }
    auto newLayerPos = mNLayer.insert(mNLayer.begin() + INPUT_LAYER_OFFSET + insertLayerBefore, NLayer(layerSz, 0));
    auto prevLayerPos = newLayerPos - 1, nextLayerPos = newLayerPos + 1;
    newLayerPos->resizeNumWeightsPerNeuron(prevLayerPos->size());
    nextLayerPos->resizeNumWeightsPerNeuron(newLayerPos->size());
    return true;
}

bool NNetwork::changeLayerSz(size_t layerPos, size_t newLayerSz)
{
    layer(layerPos).resizeLayer(newLayerSz);
    auto nextLayer = mNLayer.begin() + INPUT_LAYER_OFFSET + layerPos + 1;
    if (nextLayer != mNLayer.end())
    {
        nextLayer->resizeNumWeightsPerNeuron(newLayerSz);
    }
}

void NNetwork::feedforward(std::function<NetNumT(NetNumT)>& actFunc)
{
    for(auto layerIt = mNLayer.begin() + INPUT_LAYER_OFFSET; layerIt != mNLayer.end(); ++layerIt)
    {
        SingleRowT prevLayerOutput = (layerIt - 1)->getOutputs();
        Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic> currentLayerWeights = layerIt->getWeights();
        layerIt->mLayerOutputs = (prevLayerOutput * currentLayerWeights) + layerIt->getBiases();
        layerIt->mLayerOutputs = layerIt->mLayerOutputs.unaryExpr(actFunc);
    }
}

NLayer& NNetwork::outputLayer()
{
    return *(mNLayer.end() - 1);
}

NetNumT NNetwork::getOutput(const LabelT& label)
{
    return outputLayer().getOutputs()[mOutputLabels[label]];
}
