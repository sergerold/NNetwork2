//
// Created by Lenovo on 10/04/2023.
//

#include <iostream>

#include "NNetwork.h"
#include "NLayer.h"

NNetwork::NNetwork(size_t inputSz, const ClassList& labels)
{
    // add input layer
    NLayer inputLayer(inputSz, 0);
    inputLayer.mLayerOutputs.resize(1, inputSz);
    mNLayer.push_back(inputLayer);

    // add output layer
    NLayer outputLayer(labels.size(), inputSz);
    mNLayer.push_back(outputLayer);

    // add output classes
    for(size_t lPos = 0; lPos < labels.size(); ++ lPos)
    {
        mOutputClasses.emplace(*std::next(labels.begin(), Eigen::Index(lPos)), lPos);
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

const SingleRowT& NNetwork::getInputs() const
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

size_t NNetwork::numLayers() const
{
    return mNLayer.size() - INPUT_LAYER_OFFSET;
}

bool NNetwork::addLayer(size_t layerSz, size_t insertLayerBefore)
{
    if (insertLayerBefore + INPUT_LAYER_OFFSET >= mNLayer.size() || layerSz <= 0)
    {
        return false;
    }
    auto newLayerPos = mNLayer.insert(mNLayer.begin() + INPUT_LAYER_OFFSET + insertLayerBefore, NLayer(layerSz, 0));
    auto prevLayerPos = newLayerPos - 1, nextLayerPos = newLayerPos + 1;
    newLayerPos->resizeNumWeightsPerNeuron(prevLayerPos->size());
    nextLayerPos->resizeNumWeightsPerNeuron(newLayerPos->size());
    return true;
}

void NNetwork::changeLayerSz(size_t layerPos, size_t newLayerSz)
{
    layer(layerPos).resizeLayer(newLayerSz);
    auto nextLayer = mNLayer.begin() + INPUT_LAYER_OFFSET + Eigen::Index(layerPos) + 1;
    if (nextLayer != mNLayer.end())
    {
        nextLayer->resizeNumWeightsPerNeuron(newLayerSz);
    }
}

NLayer& NNetwork::outputLayer()
{
    return *(mNLayer.end() - 1);
}

NetNumT NNetwork::getOutput(const ClassT& c) const
{
    size_t outputPos = mOutputClasses.at(c);
    return mNLayer[mNLayer.size() - 1].getOutputs()[outputPos];
}

const std::map<ClassT, size_t>& NNetwork::classes() const
{
    return mOutputClasses;
}

void NNetwork::feedforward(const ActFuncList& actFuncs)
{
    if (actFuncs.size() != numLayers())
    {
        throw std::logic_error("Number of activation functions does not match layers in network");
    }
    for(size_t layerPos = 0 + INPUT_LAYER_OFFSET; layerPos < mNLayer.size(); ++layerPos)
    {
        const SingleRowT& prevLayerOutput = mNLayer[ layerPos - 1].getOutputs();
        const Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic>& currentLayerWeights = mNLayer[layerPos].getWeights();
        mNLayer[layerPos].mLayerOutputs = (prevLayerOutput * currentLayerWeights) + mNLayer[layerPos].getBiases();
        applyActFuncToLayer(mNLayer[layerPos].mLayerOutputs, actFuncs[layerPos - 1]);

        if (!mNLayer[layerPos].mLayerOutputs.array().allFinite())
        {
            throw std::logic_error("INF or NaN");
        }
    }
}

void NNetwork::debugFF(const ActFuncList& actFuncs)
{
    if (actFuncs.size() != numLayers())
    {
        throw std::logic_error("Number of activation functions does not match layers in network");
    }
    for(size_t layerPos = 0 + INPUT_LAYER_OFFSET; layerPos < mNLayer.size(); ++layerPos)
    {
        const SingleRowT& prevLayerOutput = mNLayer[ layerPos - 1].getOutputs();
        const Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic>& currentLayerWeights = mNLayer[layerPos].getWeights();
        mNLayer[layerPos].mLayerOutputs = (prevLayerOutput * currentLayerWeights) + mNLayer[layerPos].getBiases();
        if (layerPos == mNLayer.size() - 1)
        {
            std::cout << "###############" << std::endl;
            std::cout << mNLayer[layerPos].mLayerOutputs;
            std::cout << "###############" << std::endl;
        }
        applyActFuncToLayer(mNLayer[layerPos].mLayerOutputs, actFuncs[layerPos - 1]);
        if (layerPos == mNLayer.size() - 1)
        {
            std::cout << "**************" << std::endl;
            std::cout << mNLayer[layerPos].mLayerOutputs;
            std::cout << "**************" << std::endl;
        }
    }
}

void NNetwork::applyActFuncToLayer(SingleRowT& netInputs, ActFunc actFunc)
{
    if (actFunc == ActFunc::SIGMOID)
    {
        netInputs = netInputs.unaryExpr([](NetNumT input) -> NetNumT {return 1/(1 + exp( -input));});
    }
    if (actFunc == ActFunc::RELU)
    {
        netInputs = netInputs.unaryExpr([](NetNumT input) ->NetNumT {return fmax(0, input);});
    }
    if (actFunc == ActFunc::SOFTMAX)
    {
        // compute normalised e^x
        auto inputsAsNormExp = (netInputs.array() - netInputs.maxCoeff()).exp();
        double expSum = inputsAsNormExp.sum();
        netInputs = inputsAsNormExp.unaryExpr([&](NetNumT input) ->NetNumT {return input / expSum;});
    }
}