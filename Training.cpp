//
// Created by Lenovo on 08/05/2023.
//

#include "Training.h"

#include <iostream>

// TYPES

WeightGradients::WeightGradients(size_t numLayers)
{
    weightGradients.insert(weightGradients.begin(), numLayers, LayerWeightsT());
}

void WeightGradients::addWeightGradientsForLayer(const LayerWeightsT& newWeightGrads, size_t layer)
{
    if(layer >= weightGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    weightGradients[layer] = newWeightGrads;
}

const LayerWeightsT& WeightGradients::getWeightGradientsForLayer(size_t layer) const
{
    if(layer >= weightGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    return weightGradients[layer];
}

size_t WeightGradients::numLayers()
{
    return weightGradients.size();
}
//***********//

LayerGradients::LayerGradients(size_t numLayers)
{
    layerGradients.insert(layerGradients.begin(), numLayers, SingleRowT());
}

void LayerGradients::addLayerGradients(const SingleRowT& newLayerGrads, size_t layer)
{
    if(layer >= layerGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    layerGradients[layer] = newLayerGrads;
}

const SingleRowT& LayerGradients::getLayerGradients(size_t layer) const
{
    if(layer >= layerGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    return layerGradients[layer];
}

size_t LayerGradients::numLayers()
{
    return layerGradients.size();
}

// DATA FUNCS

SingleRowT trainingItemToVector(const std::map<ClassT, NetNumT>& trItem)
{
    size_t count = 0;
    SingleRowT targetValuesAsVector;
    targetValuesAsVector.resize(Eigen::Index(trItem.size()));
    for(auto & it : trItem)
    {
        targetValuesAsVector(0, Eigen::Index (count) ) = it.second;
        count++;
    }
    return targetValuesAsVector;
}

bool isTrainingDataValid(const std::map<ClassT, size_t>& networkLabels, const TrainingData& trainingData, size_t networkInputSz)
{
    for(const auto& item : trainingData)
    {
        const Labels& targets = item.labels;
        if (targets.size() != networkLabels.size())
        {
            return false;
        }
        for (const auto & networkLabel : networkLabels)
        {
            if (targets.count(networkLabel.first) < 1)
            {
                return false;
            }
        }
        if (item.inputs.size() != networkInputSz)
        {
            return false;
        }
    }
    return true;
}

// LOSS FUNCTIONS


NetNumT calculateLoss(const Labels& labels, LossFunc lossFunc, const SingleRowT& networkOut)
{
    SingleRowT targetValuesAsVector = trainingItemToVector(labels);
    if (lossFunc == LossFunc::MSE)
    {
        return (networkOut - targetValuesAsVector).array().square().sum() / (NetNumT) targetValuesAsVector.size();
    }
    if (lossFunc == LossFunc::CROSS_ENTROPY)
    {
        return -(networkOut.array().log() * targetValuesAsVector.array()).sum();
    }
}

// GRADIENT CALCULATION ALGORITHMS
void initialiseWeightsBiases(NNetwork& network, InitMethod method, const ActFuncList & actFuncs)
{
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        LayerWeightsT lWeights = network.layer(layerPos).getWeights();
        SingleRowT lBiases = network.layer(layerPos).getBiases();
        if (method == InitMethod::RANDOM) // random values between -1 and 1
        {
            lWeights.setRandom();
            lBiases.setRandom();
        }

        network.layer(layerPos).setWeights(lWeights);
        network.layer(layerPos).setBiases(lBiases);
    }
}

SingleRowT calculateActivationFunctionGradients(const NLayer& layer, ActFunc actFunc)
{
    if(actFunc == ActFunc::SIGMOID)
    {
        return layer.getOutputs().array() * (1 - layer.getOutputs().array());
    }
    if(actFunc == ActFunc::RELU)
    {
        // Heaviside step function (derivative undefined at input 0 so set at 0.5)
        return (layer.getOutputs().array().sign() + 1) * 0.5;
    }
}

SingleRowT calculateOutputLayerGradientsForTrainingItem(const NLayer& outputLayer, ActFunc actFuncForLayer, LossFunc lossFunc, const Labels& targets)
{
    // convert target to vector format
    SingleRowT targetsVec = trainingItemToVector(targets);
    // calculate layer gradients based upon loss function
    if (lossFunc == LossFunc::CROSS_ENTROPY)
    {
        // simplified calculation of derivative for cross entropy loss and softmax activation
        return outputLayer.getOutputs() - targetsVec;
    }
    if (lossFunc == LossFunc::MSE)
    {
        SingleRowT gradientOfMse = outputLayer.getOutputs() - targetsVec;
        SingleRowT gradientOfActFunc = calculateActivationFunctionGradients(outputLayer, actFuncForLayer);
        return gradientOfMse.array() * gradientOfActFunc.array();
    }
}

void calculateHiddenLayerGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LayerGradients& layerGrads)
{
    size_t lastHiddenLayer = network.numLayers() - 2; // -1 is the output layer so -2 is last hidden layer
    for (size_t layerPos = lastHiddenLayer; ;--layerPos) // reverse backwards through each layer
    {
        const LayerWeightsT& weightsOfSubsequentLayer = network.layer(layerPos + 1).getWeights();
        const SingleRowT& subsequentLayerError = layerGrads.getLayerGradients(layerPos + 1);
        SingleRowT errorWrtOutput = subsequentLayerError * weightsOfSubsequentLayer.transpose();
        SingleRowT activationFunctionGradient = calculateActivationFunctionGradients(network.layer(layerPos), actFuncs[layerPos]);
        SingleRowT errorWrtNetInput = errorWrtOutput.array() * activationFunctionGradient.array();
        layerGrads.addLayerGradients(errorWrtNetInput, layerPos);

        if (layerPos == 0)
        {
            break;
        }
    }
}

void calculateWeightGradientsForTrainingItem(NNetwork& network, const LayerGradients& layerGrads, WeightGradients& weightGrads)
{
    size_t outputLayer = network.numLayers() - 1;
    for(size_t layerPos = outputLayer; ; --layerPos)
    {
        SingleRowT prevLayerOutput;
        if(layerPos > 0)
        {
            prevLayerOutput = network.layer(layerPos - 1).getOutputs();
        }
        else
        {
            prevLayerOutput = network.getInputs(); // if layer is first hidden layer then output of previous layer is input
        }
        const SingleRowT& currentLayerGrad = layerGrads.getLayerGradients(layerPos);

        LayerWeightsT outerProduct = prevLayerOutput.transpose() * currentLayerGrad;
        weightGrads.addWeightGradientsForLayer(outerProduct, layerPos);

        if(layerPos == 0)
        {
            break;
        }
    }
}

void calculateGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LossFunc lossFunc, TrainingItem& trItem, LayerGradients& layerGrads, WeightGradients& weightGrads)
{
    if(actFuncs.size() != network.numLayers())
    {
        throw std::logic_error("List of activation functions does not equal number of layers");
    }
    if(lossFunc == LossFunc::CROSS_ENTROPY && actFuncs[network.numLayers() - 1] != ActFunc::SOFTMAX)
    {
        throw std::logic_error("If cross entropy loss function then final hidden layer must use softmax activation function");
    }
    // load inputs and feedforward
    network.setInputs(trItem.inputs);
    network.feedforward(actFuncs);

    // calculate the final layer gradients
    size_t outputLayerPos = network.numLayers() - 1;
    SingleRowT outputLayerGradients = calculateOutputLayerGradientsForTrainingItem(network.outputLayer(), actFuncs[outputLayerPos], lossFunc, trItem.labels);
    layerGrads.addLayerGradients(outputLayerGradients, outputLayerPos);

    // calculate the hidden layer gradients
    calculateHiddenLayerGradientsForTrainingItem(network, actFuncs,layerGrads);

    // calculate the weight gradients
    calculateWeightGradientsForTrainingItem(network, layerGrads, weightGrads);
}

void updateNetworkWeightsBiasesWithGradients(NNetwork& network, const LayerGradients& layerGrads, const WeightGradients& weightGrads, const LearningRateList& learningRatesPerLayer)
{
    if(learningRatesPerLayer.size() != network.numLayers())
    {
        throw std::logic_error("Number of learning rate layers does not match number of network layers");
    }
    size_t outputLayer = network.numLayers() - 1;
    for(size_t layerPos = outputLayer; ;--layerPos)
    {
        NetNumT learningRateForLayer = learningRatesPerLayer[layerPos];
        // update weights
        const LayerWeightsT& layerWeights = network.layer(layerPos).getWeights();
        const LayerWeightsT& layerWeightGrads = weightGrads.getWeightGradientsForLayer(layerPos);
        LayerWeightsT updatedWeights = layerWeights - (layerWeightGrads * learningRateForLayer);
        network.layer(layerPos).setWeights(updatedWeights);
        // update biases
        const SingleRowT& layerBiases = network.layer(layerPos).getBiases();
        const SingleRowT& layerBiasGrads = layerGrads.getLayerGradients(layerPos);
        SingleRowT updatedBiases = layerBiases - (layerBiasGrads * learningRateForLayer);
        network.layer(layerPos).setBiases(updatedBiases);

        if(layerPos == 0)
        {
            break;
        }
    }
}