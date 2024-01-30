//
// Created by Lenovo on 08/05/2023.
//

#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>


#include "Training.h"
#include "Data.h"
//#include "Debug.h"

// TYPES

WeightGradients::WeightGradients(size_t numLayers)
{
    weightGradients.insert(weightGradients.begin(), numLayers, LayerWeightsT());
}

WeightGradients::WeightGradients(NNetwork& network)
{
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        weightGradients.push_back(network.layer(layerPos).getWeights());
        weightGradients[layerPos].setZero();
    }
}

void WeightGradients::insertWeightGradientsForLayer(const LayerWeightsT& newWeightGrads, size_t layer)
{
    if(layer >= weightGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    weightGradients[layer] = (newWeightGrads);
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
        weightGradients[layer].noalias() += weightsToAdd.getWeightGradientsForLayer(layer); // efficient way to add
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

LayerGradients::LayerGradients(NNetwork& network)
{
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        layerGradients.push_back(network.layer(layerPos).getBiases());
        layerGradients[layerPos].setZero();
    }

}

void LayerGradients::insertLayerGradients(const SingleRowT& newLayerGrads, size_t layer)
{
    if(layer >= layerGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    layerGradients[layer] = (newLayerGrads);
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
        layerGradients[layerPos].noalias() += layerGradsToAdd.getLayerGradients(layerPos); // efficient way to add
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

NetNumT calculateLossForExampleItem(const Labels& labels, LossFunc lossFunc, const SingleRowT& networkOut)
{
    if (lossFunc == LossFunc::MSE)
    {
        return (networkOut - labels).array().square().sum() / static_cast<NetNumT> (labels.size());
    }
    else if (lossFunc == LossFunc::CROSS_ENTROPY)
    {
        constexpr NetNumT VERY_SMALL_NUMBER = 0.0000001f; // add this to output values so as to ensure no log(0)
        return -( (networkOut.array() + VERY_SMALL_NUMBER).log() * labels.array()).sum();
    }
    else
    {
        throw std::runtime_error("Unsupported loss function.");
    }
}

NetNumT calculateLossForExampleData(NNetwork& network, const ExampleData& trData, const ActFuncList& actFuncs, LossFunc lossFunc)
{
    NetNumT trainingError = 0;
    for(const ExampleItem& trItem : trData)
    {
        network.setInputs(trItem.inputs.cast<NetNumT>());

        network.feedforward(actFuncs, 0);
        trainingError += calculateLossForExampleItem(trItem.labels, lossFunc, network.outputLayer().getOutputs());
    }
    return trainingError / static_cast<NetNumT> (trData.size()); // return average
}

NetNumT calculateAccuracyForExampleData(NNetwork& network, const ExampleData& data, const ActFuncList &actFuncs)
{
    NetNumT correct = 0;
    for(const auto& item : data)
    {
        network.setInputs(item.inputs);
        network.feedforward(actFuncs, 0);
        if(actFuncs[actFuncs.size() - 1] == ActFunc::SOFTMAX)
        {
            const SingleRowT& output = network.outputLayer().getOutputs();
            // find highest probability in output
            const auto highestElementIt = std::max_element(output.begin(), output.end());
            const Eigen::Index posOfHighestElement = std::distance(output.begin(), highestElementIt);

            if(item.labels.coeff(0, posOfHighestElement) == 1)
            {
                correct++;
            }
        }
    }
    return (correct / static_cast<NetNumT> (data.size()) * 100);
}

// GRADIENT CALCULATION ALGORITHMS
void initialiseWeightsBiases(NNetwork& network, InitMethod method)
{
    if(method == InitMethod::NO_INIT)
    {
        // if no init then leave weights and biases
        return;
    }
    std::default_random_engine generator(12345);
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
            const Eigen::Index prevLayerSz = lWeights.rows();
            const NetNumT mean = 0, sd = static_cast<NetNumT> ( sqrt( (2.0/ static_cast<NetNumT> (prevLayerSz) )) );
            std::normal_distribution<NetNumT> distribution(mean,sd);
            lWeights = LayerWeightsT::NullaryExpr(lWeights.rows(), lWeights.cols(),[&](){return distribution(generator);});
        }
        if(method == InitMethod::UNIFORM_HE)
        {
            const Eigen::Index prevLayerSz = lWeights.rows(), nextLayerSz = lWeights.cols();
            const auto lowerBound = static_cast<NetNumT> ( -(sqrt(6.0/ static_cast<NetNumT> (prevLayerSz + nextLayerSz))) );
            const auto upperBound = static_cast<NetNumT> ( sqrt(6.0/ static_cast<NetNumT> (prevLayerSz + nextLayerSz)) );
            std::uniform_real_distribution<NetNumT> distribution(lowerBound, upperBound);
            lWeights = LayerWeightsT::NullaryExpr(lWeights.rows(), lWeights.cols(),[&](){return distribution(generator);});
        }
        if(method == InitMethod::NORMALISED_XAVIER)
        {
            const Eigen::Index prevLayerSz = lWeights.rows(), nextLayerSz = lWeights.cols();
            const NetNumT mean = 0, sd = static_cast<NetNumT> (sqrt(2.0/(static_cast<NetNumT> (prevLayerSz + nextLayerSz))) );
            std::normal_distribution<NetNumT> distribution(mean,sd);
            lWeights = LayerWeightsT::NullaryExpr(lWeights.rows(), lWeights.cols(),[&](){return distribution(generator);});
        }
        if(method == InitMethod::UNIFORM_XAVIER)
        {
            const Eigen::Index prevLayerSz = lWeights.rows(), nextLayerSz = lWeights.cols();
            const auto lowerBound = static_cast<NetNumT> (-sqrt(6.0/static_cast<NetNumT> ( prevLayerSz + nextLayerSz)) );
            const auto upperBound = static_cast<NetNumT> (sqrt(6.0/static_cast<NetNumT>(prevLayerSz + nextLayerSz )) );
            std::uniform_real_distribution<NetNumT> distribution(lowerBound, upperBound);
            lWeights = LayerWeightsT::NullaryExpr(lWeights.rows(), lWeights.cols(),[&](){return distribution(generator);});
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
    else if(actFunc == ActFunc::RELU)
    {
        // Heaviside step function (derivative undefined at input 0 so set at 0)
        return (layer.getOutputs().array() > 0).template cast<NetNumT>();
    }
    // no softmax derivative as always combined with cross entropy loss
    else {
        throw std::runtime_error("Unsupported loss function.");
    }
}

SingleRowT calculateOutputLayerGradientsForTrainingItem(const NLayer& outputLayer, ActFunc actFuncForLayer, LossFunc lossFunc, const Labels& targets)
{
    // convert target to vector format
    // calculate layer gradients based upon loss function
    if (lossFunc == LossFunc::CROSS_ENTROPY)
    {
        // simplified calculation of derivative for cross entropy loss and softmax activation
        return outputLayer.getOutputs() - targets;
    }
    if (lossFunc == LossFunc::MSE)
    {
        SingleRowT gradientOfMse = outputLayer.getOutputs() - targets;
        SingleRowT gradientOfActFunc = calculateActivationFunctionGradients(outputLayer, actFuncForLayer);
        return gradientOfMse.array() * gradientOfActFunc.array();
    }
    else {
        throw std::runtime_error("Unsupported loss function.");
    }
}

void calculateHiddenLayerGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LayerGradients& layerGrads)
{
    const size_t lastHiddenLayer = network.numLayers() - 2; // -1 is the output layer so -2 is last hidden layer
    for (size_t layerPos = lastHiddenLayer; ;--layerPos) // reverse backwards through each layer
    {
        const LayerWeightsT& weightsOfSubsequentLayer = network.layer(layerPos + 1).getWeights();
        const SingleRowT& subsequentLayerError = layerGrads.getLayerGradients(layerPos + 1);
        SingleRowT errorWrtOutput = subsequentLayerError * weightsOfSubsequentLayer.transpose();
        SingleRowT activationFunctionGradient = calculateActivationFunctionGradients(network.layer(layerPos), actFuncs[layerPos]);
        SingleRowT errorWrtNetInput = errorWrtOutput.array() * activationFunctionGradient.array();

        if (!errorWrtNetInput.allFinite())
        {
            throw std::logic_error("(1) Contains INF or NaN");
        }
        layerGrads.insertLayerGradients(errorWrtNetInput, layerPos);

        if (layerPos == 0)
        {
            break;
        }
    }
}

void calculateWeightGradientsForTrainingItem(NNetwork& network, const LayerGradients& layerGrads, WeightGradients& weightGrads)
{
    const size_t outputLayer = network.numLayers() - 1;
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
        if (!outerProduct.allFinite())
        {
            throw std::logic_error("(2) Contains INF or NaN");
        }
        weightGrads.insertWeightGradientsForLayer(outerProduct, layerPos);

        // ugly way of terminating loop when moving back
        if(layerPos == 0)
        {
            break;
        }
    }
}

void calculateGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LossFunc lossFunc, const ExampleItem& trItem, LayerGradients& layerGrads, WeightGradients& weightGrads, NetNumT dropOutRate)
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

    network.feedforward(actFuncs, dropOutRate);

    // calculate the final layer gradients
    const size_t outputLayerPos = network.numLayers() - 1;
    const SingleRowT outputLayerGradients = calculateOutputLayerGradientsForTrainingItem(network.outputLayer(), actFuncs[outputLayerPos], lossFunc, trItem.labels);
    if(!outputLayerGradients.allFinite())
    {
        throw std::logic_error("(3) Contains INF or NaN");
    }
    layerGrads.insertLayerGradients(outputLayerGradients, outputLayerPos);

    // calculate the hidden layer gradients
    calculateHiddenLayerGradientsForTrainingItem(network, actFuncs,layerGrads);

    // calculate the weight gradients
    calculateWeightGradientsForTrainingItem(network, layerGrads, weightGrads);


}

void updateNetworkWeightsBiasesWithGradients(NNetwork& network, const LayerGradients& layerGrads, const WeightGradients& weightGrads, const LearningRateList& learningRatesPerLayer, NetNumT momentumFactor, LayerGradients& prevUpdateBiasDelta, WeightGradients& prevUpdateWeightDelta)
{
    if(learningRatesPerLayer.size() != network.numLayers())
    {
        throw std::logic_error("Number of learning rate layers does not match number of network layers");
    }
    const size_t outputLayer = network.numLayers() - 1;
    for(size_t layerPos = outputLayer; ;--layerPos)
    {
        const NetNumT learningRateForLayer = learningRatesPerLayer[layerPos];

        // update weights
        const LayerWeightsT& layerWeights = network.layer(layerPos).getWeights();
        const LayerWeightsT& layerWeightGrads = weightGrads.getWeightGradientsForLayer(layerPos);
        const LayerWeightsT& layerWeightsDelta = ((1.0 - momentumFactor) * layerWeightGrads) + (momentumFactor * prevUpdateWeightDelta.getWeightGradientsForLayer(layerPos));
        network.layer(layerPos).setWeights(layerWeights - (learningRateForLayer * layerWeightsDelta));
        prevUpdateWeightDelta.insertWeightGradientsForLayer(layerWeightsDelta, layerPos);

        // update biases
        const SingleRowT& layerBiases = network.layer(layerPos).getBiases();
        const SingleRowT& layerBiasGrads = layerGrads.getLayerGradients(layerPos);
        const SingleRowT& layerBiasDelta = ((1 - momentumFactor) * layerBiasGrads) + (momentumFactor * prevUpdateBiasDelta.getLayerGradients(layerPos));
        network.layer(layerPos).setBiases(layerBiases - (learningRateForLayer * layerBiasDelta));
        prevUpdateBiasDelta.insertLayerGradients(layerBiasDelta, layerPos);

        if(layerPos == 0)
        {
            break;
        }
    }
}

void calculateGradientsOverBatch(NNetwork& network, ExampleData::iterator batchStart, ExampleData::iterator batchEnd, const ActFuncList& actFuncs, LossFunc lossFunc, LayerGradients& layerGrads, WeightGradients& weightGrads, NetNumT dropOutRate)
{
    size_t count = 0;
    for(auto trItemIt = batchStart; trItemIt != batchEnd; ++trItemIt)
    {
        LayerGradients layerGradientsForTrItem(network.numLayers());
        WeightGradients weightGradientsForTrItem(network.numLayers());
        calculateGradientsForTrainingItem(network, actFuncs, lossFunc, *trItemIt, layerGradientsForTrItem, weightGradientsForTrItem, dropOutRate);
        layerGrads.addLayerGradients(layerGradientsForTrItem);
        weightGrads.addWeightGradients(weightGradientsForTrItem);
        ++count;
    }
    layerGrads.divideLayerGradients(std::distance(batchStart, batchEnd));
    weightGrads.divideWeightGradients(std::distance(batchStart, batchEnd));
}

void train(NNetwork& network, ExampleData& trainingData, const ActFuncList& actFuncs, LossFunc lossFunc, const LearningRateList& lrList, NetNumT momentum, InitMethod initMethod, size_t epochsToRun, size_t batchSz, const ExampleData& testData, NetNumT dropOutRate)
{
    if (!isTrainingDataValid(network.classes(), trainingData, network.getInputs().size()))
    {
        throw std::logic_error("Training data invalid");
    }
    initialiseWeightsBiases(network, initMethod);

    // prev weight updates for momentum - set to 0 for first update
    WeightGradients prevWeightDelta(network);
    LayerGradients prevBiasDelta(network);

    // each epoch

    for(size_t epoch = 0; epoch < epochsToRun; ++epoch)
    {
        auto start = std::chrono::steady_clock::now();

        // random shuffle and then update for each minibatch
        std::shuffle(trainingData.begin(), trainingData.end(), std::default_random_engine(12345));
        for(auto trItemIt = trainingData.begin(); trItemIt < trainingData.end(); trItemIt += static_cast<std::vector<ExampleItem>::difference_type>(batchSz))
        {
            auto batchEnd = trItemIt + static_cast<std::vector<ExampleItem>::difference_type>(batchSz);
            if(batchEnd > trainingData.end())
            {
                batchEnd = trainingData.end();
            }
            LayerGradients lGradsOverBatch(network.numLayers());
            WeightGradients wGradsOverBatch(network.numLayers());
            calculateGradientsOverBatch(network, trItemIt, batchEnd, actFuncs, lossFunc, lGradsOverBatch, wGradsOverBatch, dropOutRate);
            updateNetworkWeightsBiasesWithGradients(network, lGradsOverBatch, wGradsOverBatch, lrList, momentum, prevBiasDelta, prevWeightDelta);
        }

        auto end = std::chrono::steady_clock::now();

        // Print

        std::cout << "Epoch: " << epoch << std::endl;
        std::cout << "Time: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;
        std::cout << " -> Training Data (" << trainingData.size() << " items):\n";
        std::cout << "   --> Average Loss: " << std::fixed << calculateLossForExampleData(network, trainingData, actFuncs, lossFunc) << std::endl;
        std::cout << "   --> Accuracy: " << std::fixed << calculateAccuracyForExampleData(network, trainingData, actFuncs) << "%" << std::endl;

        std::cout << " -> Test Data (" << testData.size() << " items):\n";
        std::cout << "   --> Average Loss: " << std::fixed << calculateLossForExampleData(network, testData, actFuncs, lossFunc) << std::endl;
        std::cout << "   --> Accuracy: " << std::fixed << calculateAccuracyForExampleData(network, testData, actFuncs) << "%" << std::endl;
        std::cout << "********************\n";

    }
}