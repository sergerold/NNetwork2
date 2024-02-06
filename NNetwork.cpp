//
// Created by Lenovo on 10/04/2023.
//

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>

#include "NNetwork.h"
#include "NLayer.h"
//#include "Debug.h"

std::default_random_engine gen ( 12345 );


NNetwork::NNetwork(size_t inputSz, const ClassList& labels)
{
    // add input layer
    NLayer inputLayer(inputSz, 0);
    inputLayer.mLayerOutputs.resize(1, static_cast<Eigen::Index> (inputSz));
    mNLayer.push_back(inputLayer);

    // add output layer
    NLayer outputLayer(labels.size(), inputSz);
    mNLayer.emplace_back(outputLayer);

    // add output classes
    for(size_t lPos = 0; lPos < labels.size(); ++ lPos)
    {
        mOutputClasses.emplace(*std::next(labels.begin(), static_cast<Eigen::Index>(lPos)), lPos);
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
    if (insertLayerBefore + INPUT_LAYER_OFFSET >= mNLayer.size())
    {
        return false;
    }
    const auto newLayerPos = mNLayer.insert(mNLayer.begin() + static_cast<Eigen::Index> (INPUT_LAYER_OFFSET + insertLayerBefore), NLayer(layerSz, 0));
    const auto prevLayerPos = newLayerPos - 1, nextLayerPos = newLayerPos + 1;
    newLayerPos->resizeNumWeightsPerNeuron(prevLayerPos->size());
    nextLayerPos->resizeNumWeightsPerNeuron(newLayerPos->size());
    return true;
}

void NNetwork::changeLayerSz(size_t layerPos, size_t newLayerSz)
{
    layer(layerPos).resizeLayer(newLayerSz);
    const auto nextLayer = mNLayer.begin() + static_cast<Eigen::Index> (INPUT_LAYER_OFFSET + layerPos + 1);
    if (nextLayer != mNLayer.end())
    {
        nextLayer->resizeNumWeightsPerNeuron(newLayerSz);
    }
}

NLayer& NNetwork::outputLayer()  {
    return *(mNLayer.end() - 1);
}

NetNumT NNetwork::getOutput(const ClassT& c) const
{
    if(mOutputClasses.count(c) == 0)
    {
        throw std::out_of_range("No such class");
    }
    const size_t outputPos = mOutputClasses.at(c);
    return mNLayer[mNLayer.size() - 1].getOutputs()[static_cast<Eigen::Index> (outputPos)];
}

const std::map<ClassT, size_t>& NNetwork::classes() const
{
    return mOutputClasses;
}

void NNetwork::feedforward(const ActFuncList& actFuncs, NetNumT dropOutRate)
{
    std::bernoulli_distribution distribution(1 - dropOutRate);
    if (actFuncs.size() != numLayers())
    {
        throw std::logic_error("Number of activation functions does not match layers in network");
    }

    // Find the maximum layer size for dropout mask pre-allocation
    Eigen::Index maxSize = 0;
    for (const auto& layer : mNLayer)
    {
        maxSize = std::max(maxSize, static_cast<Eigen::Index>(layer.size()));
    }
    SingleRowT dropOutMask(1, maxSize);

    // starting at the first hidden layer and then moving to the output layer...
    for(size_t layerPos = 0 + INPUT_LAYER_OFFSET; layerPos < mNLayer.size(); ++layerPos)
    {
        auto& layer = mNLayer[layerPos]; // current layer
        const SingleRowT& prevLayerOutput = mNLayer[layerPos - 1].getOutputs(); // outputs from previous layer
        layer.mLayerOutputs.noalias() = (prevLayerOutput * layer.getWeights()) +  layer.getBiases().row(0); // calculate matrix multiplication and add biases

        // apply drop out
        if (layerPos < mNLayer.size() - 1 && dropOutRate > 0) // Except the output layer
        {
            dropOutMask.leftCols(layer.size()).setZero();
            for(Eigen::Index maskPos = 0; maskPos < layer.size(); ++maskPos)
            {
                dropOutMask(0, maskPos) = distribution(gen);
            }
            layer.mLayerOutputs.array() *= dropOutMask.leftCols(layer.size()).array();
            layer.mLayerOutputs.array() /= (1 - dropOutRate);
        }
        // apply activation function
        applyActFuncToLayer(layer.mLayerOutputs, actFuncs[layerPos - 1]);

        if (!layer.mLayerOutputs.allFinite())
        {
            throw std::logic_error("(5) INF or NaN");
        }
    }
}

void NNetwork::applyActFuncToLayer(SingleRowT& netInputs, ActFunc actFunc)
{
    if (netInputs.size() < 1) {
        throw std::logic_error("Size of netinputs is 0");
    }

    switch (actFunc) {
        case ActFunc::SIGMOID: {
            netInputs = 1.0 / (1.0 + (-netInputs.array()).exp());
            break;
        }

        case ActFunc::RELU: {
            netInputs = netInputs.cwiseMax(0);
            break;
        }

        case ActFunc::SOFTMAX: {
            // compute normalised e^x
            const auto maxCoeff = netInputs.maxCoeff();
            if (std::isnan(maxCoeff) || std::isinf(maxCoeff)) {
                throw std::logic_error("Max coefficient is NaN or INF");
            }
            const auto inputsAsNormExp = (netInputs.array() - maxCoeff).exp();
            const double expSum = inputsAsNormExp.sum();
            netInputs = inputsAsNormExp / expSum ;
            break;
        }

        default:
            throw std::runtime_error("Unsupported activation function");
    }
}

std::ostream& NNetwork::summarise(std::ostream& printer)
{
    printer << "*******************\nNETWORK SUMMARY\n*******************" << std::endl;
    printer << "Input size: " << getInputs().size() << std::endl;
    printer << "Number of Layers: " << numLayers() << std::endl;
    printer << "*******************" << std::endl;
    unsigned int weightCount = 0, biasCount = 0;
    for(size_t layerPos = 0; layerPos < numLayers(); ++layerPos)
    {
        printer << "Layer " << layerPos << ", size : " << layer(layerPos).getOutputs().size() << std::endl;
        printer << "Weight Dimensions " << " : Rows (" << layer(layerPos).getWeights().rows() << ")" << ", " << "Cols (" << layer(layerPos).getWeights().cols() << ")" << std::endl;
        printer << "+++++++++" << std::endl;
        biasCount += layer(layerPos).getOutputs().size();
        weightCount += layer(layerPos).getWeights().rows() * layer(layerPos).getWeights().cols();
    }
    printer << "Bias count: " << biasCount << "\nWeight Count: " << weightCount << std::endl;
    printer << "*******************" << std::endl;
    printer << "Classes: " << std::endl;
    for(auto & mOutputClasse : mOutputClasses)
    {
        printer << "--> \"" << mOutputClasse.first << "\"" << std::endl;
    }
    return printer;
}