//
// Created by Lenovo on 08/05/2023.
//

#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>


#include "Training.h"
#include "Data.h"

// TYPES

WeightGradients::WeightGradients(size_t numLayers)
{
    weightGradients.insert(weightGradients.begin(), numLayers, LayerWeightsT());
}

void WeightGradients::insertWeightGradientsForLayer(const LayerWeightsT& newWeightGrads, size_t layer)
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

void WeightGradients::addWeightGradients(const WeightGradients& weightsToAdd)
{
    if (weightsToAdd.numLayers() != numLayers())
    {
        throw std::logic_error("Add: Number of Weight Gradients do not match");
    }
    for(size_t layer = 0; layer < numLayers(); ++layer)
    {
        // if weights is empty then simply assign weightsToAdd
        if(weightGradients[layer].size() == 0)
        {
            weightGradients[layer] = weightsToAdd.getWeightGradientsForLayer(layer);
            continue;
        }
        // if weights do not match then throw error
        if(weightGradients[layer].rows() != weightsToAdd.getWeightGradientsForLayer(layer).rows() ||
           weightGradients[layer].cols() != weightsToAdd.getWeightGradientsForLayer(layer).cols())
        {
            throw std::out_of_range("Weight dimensions do not match");
        }
        weightGradients[layer] = weightGradients[layer] + weightsToAdd.getWeightGradientsForLayer(layer);
    }
}

void WeightGradients::divideWeightGradients(size_t divideBy)
{
    for(size_t layerPos = 0; layerPos < numLayers(); ++layerPos)
    {
        weightGradients[layerPos] = weightGradients[layerPos].array() / divideBy;
    }
}

size_t WeightGradients::numLayers() const
{
    return weightGradients.size();
}

//***********//

LayerGradients::LayerGradients(size_t numLayers)
{
    layerGradients.insert(layerGradients.begin(), numLayers, SingleRowT());
}

void LayerGradients::insertLayerGradients(const SingleRowT& newLayerGrads, size_t layer)
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

size_t LayerGradients::numLayers() const
{
    return layerGradients.size();
}

void LayerGradients::addLayerGradients(const LayerGradients& layerGradsToAdd)
{
    if(numLayers() != layerGradsToAdd.numLayers())
    {
        throw std::out_of_range("Add: Number of layers do not match");
    }
    for(size_t layerPos = 0; layerPos < numLayers(); ++layerPos)
    {
        // if layer is empty then simply assign layerGradsToAdd
        if(layerGradients[layerPos].size() == 0)
        {
            layerGradients[layerPos] = layerGradsToAdd.getLayerGradients(layerPos);
            continue;
        }
        // throw error if layers of different size
        if ( layerGradients[layerPos].size() != layerGradsToAdd.getLayerGradients(layerPos).size() )
        {
            throw std::out_of_range("Layer dimensions do not match");
        }
        // add
        layerGradients[layerPos] = layerGradients[layerPos] + layerGradsToAdd.getLayerGradients(layerPos);
    }
}

void LayerGradients::divideLayerGradients(size_t divideBy)
{
    for(size_t layerPos = 0; layerPos < numLayers(); ++layerPos)
    {
        layerGradients[layerPos] = layerGradients[layerPos].array() / divideBy;
    }

}

// LOSS FUNCTIONS


NetNumT calculateLossForTrainingItem(const Labels& labels, LossFunc lossFunc, const SingleRowT& networkOut)
{
    SingleRowT targetValuesAsVector = trainingItemToVector(labels);
    if (lossFunc == LossFunc::MSE)
    {
        return (networkOut - targetValuesAsVector).array().square().sum() / (NetNumT) targetValuesAsVector.size();
    }
    if (lossFunc == LossFunc::CROSS_ENTROPY)
    {
        const NetNumT VERY_SMALL_NUMBER = 0.0000001; // add this to output values so as to ensure no log(0)
        return -( (networkOut.array() + VERY_SMALL_NUMBER).log() * targetValuesAsVector.array()).sum();
    }
}

NetNumT calculateLossForTrainingData(NNetwork& network, const TrainingData& trData, const ActFuncList& actFuncs, LossFunc lossFunc)
{
    NetNumT trainingError = 0;
    for(const TrainingItem& trItem : trData)
    {
        network.setInputs(trItem.inputs.cast<NetNumT>());

        network.feedforward(actFuncs);
        trainingError += calculateLossForTrainingItem(trItem.labels, lossFunc, network.outputLayer().getOutputs());
    }
    return trainingError / (NetNumT) trData.size(); // return average
}

// GRADIENT CALCULATION ALGORITHMS
void initialiseWeightsBiases(NNetwork& network, InitMethod method, const ActFuncList & actFuncs)
{
    std::default_random_engine generator;
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        LayerWeightsT lWeights = network.layer(layerPos).getWeights();
        SingleRowT lBiases = network.layer(layerPos).getBiases();
        if (method == InitMethod::RANDOM_UNIFORM) // random values between -1 and 1
        {
            lWeights.setRandom();
        }
        if(method == InitMethod::NORMALISED_HE)
        {
            NetNumT prevLayerSz = lWeights.rows();
            NetNumT mean = 0, sd = sqrt(2/prevLayerSz);
            std::normal_distribution<double> distribution(mean,sd);
            lWeights = lWeights.unaryExpr([&](NetNumT wValue) ->NetNumT {return distribution(generator);});
        }
        if(method == InitMethod::UNIFORM_HE)
        {
            NetNumT prevLayerSz = lWeights.rows(), nextLayerSz = lWeights.cols();
            NetNumT lowerBound = -(sqrt(6/(prevLayerSz + nextLayerSz))), upperBound = sqrt(6/(prevLayerSz + nextLayerSz));
            std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
            lWeights = lWeights.unaryExpr([&](NetNumT wValue)->NetNumT {return distribution(generator);});
        }
        if(method == InitMethod::NORMALISED_XAVIER)
        {
            NetNumT prevLayerSz = lWeights.rows(), nextLayerSz = lWeights.cols();
            NetNumT mean = 0, sd = sqrt(2/(prevLayerSz + nextLayerSz));
            std::normal_distribution<double> distribution(mean,sd);
            lWeights = lWeights.unaryExpr([&](NetNumT wValue) ->NetNumT {return distribution(generator);});
        }
        if(method == InitMethod::UNIFORM_XAVIER)
        {
            NetNumT prevLayerSz = lWeights.rows(), nextLayerSz = lWeights.cols();
            NetNumT lowerBound = -sqrt(6/(prevLayerSz + nextLayerSz)), upperBound = sqrt(6/(prevLayerSz + nextLayerSz));
            std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
            lWeights = lWeights.unaryExpr([&](NetNumT wValue)->NetNumT {return distribution(generator);});
        }

        lBiases.setZero(); // biases tend to be intialised to zero
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
        layerGrads.insertLayerGradients(errorWrtNetInput, layerPos);

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
        weightGrads.insertWeightGradientsForLayer(outerProduct, layerPos);

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
    network.setInputs(trItem.inputs.cast<NetNumT>());
    network.feedforward(actFuncs);

    // calculate the final layer gradients
    size_t outputLayerPos = network.numLayers() - 1;
    SingleRowT outputLayerGradients = calculateOutputLayerGradientsForTrainingItem(network.outputLayer(), actFuncs[outputLayerPos], lossFunc, trItem.labels);
    layerGrads.insertLayerGradients(outputLayerGradients, outputLayerPos);

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

void calculateGradientsOverBatch(NNetwork& network, TrainingData& trData, TrainingData::iterator batchStart, size_t batchSz, const ActFuncList& actFuncs, LossFunc lossFunc, LayerGradients& layerGrads, WeightGradients& weightGrads)
{
    if(std::distance(batchStart, trData.end()) < batchSz) // shorten batch sz if remaining training items less than batch sz
    {
        batchSz = std::distance(batchStart, trData.end());
    }
    for(auto trItemIt = batchStart; trItemIt != batchStart + batchSz; ++trItemIt)
    {
        LayerGradients layerGradientsForTrItem(network.numLayers());
        WeightGradients weightGradientsForTrItem(network.numLayers());
        calculateGradientsForTrainingItem(network, actFuncs, lossFunc, *trItemIt, layerGradientsForTrItem, weightGradientsForTrItem);
        layerGrads.addLayerGradients(layerGradientsForTrItem);
        weightGrads.addWeightGradients(weightGradientsForTrItem);
    }
    layerGrads.divideLayerGradients(batchSz);
    weightGrads.divideWeightGradients(batchSz);
}

void train(NNetwork& network, TrainingData& trData, const ActFuncList& actFuncs, LossFunc lossFunc, const LearningRateList& lrList, InitMethod initMethod, size_t epochsToRun, size_t batchSz)
{
    if (!isTrainingDataValid(network.classes(), trData, network.getInputs().size()))
    {
        throw std::logic_error("Training data invalid");
    }
    initialiseWeightsBiases(network, initMethod, actFuncs);

    for(size_t epoch = 0; epoch < epochsToRun; ++epoch)
    {
        std::shuffle(trData.begin(), trData.end(), std::default_random_engine(12345));
        for(auto trItemIt = trData.begin(); trItemIt < trData.end(); trItemIt += batchSz)
        {
            LayerGradients lGradsOverBatch(network.numLayers());
            WeightGradients wGradsOverBatch(network.numLayers());
            calculateGradientsOverBatch(network, trData, trItemIt, batchSz,actFuncs, lossFunc,lGradsOverBatch,wGradsOverBatch);
            updateNetworkWeightsBiasesWithGradients(network,lGradsOverBatch, wGradsOverBatch, lrList);
        }
        std::cout << "Epoch: " << epoch << " Error: " << std::fixed << calculateLossForTrainingData(network, trData, actFuncs, lossFunc) << std::endl;
    }
}