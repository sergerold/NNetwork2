//
// Created by Lenovo on 08/05/2023.
//

#include <iostream>
#include <random> // to generate random wei
#include <algorithm>
#include <fstream> // for serialisation
#include <chrono> // for timing

#include "Training.h"
#include "Data.h"

// TYPES

NetworkWeightGradients::NetworkWeightGradients(NNetwork& network)
{
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        weightGradients.push_back(network.layer(layerPos).getWeights());
        weightGradients[layerPos].setZero();
    }
}

void NetworkWeightGradients::setWeightGradientsForLayer(const LayerWeightsT& newWeightGrads, size_t layer)
{
    if(layer >= weightGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    weightGradients[layer] = (newWeightGrads);
}

const LayerWeightsT& NetworkWeightGradients::getWeightGradientsForLayer(size_t layer) const
{
    if(layer >= weightGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    return weightGradients[layer];
}

void NetworkWeightGradients::numericAddWeightGradients(const NetworkWeightGradients& weightsToAdd)
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

void NetworkWeightGradients::divideWeightGradients(size_t divideBy)
{
    for(size_t layerPos = 0; layerPos < numLayers(); ++layerPos)
    {
        weightGradients[layerPos] = weightGradients[layerPos].array() / divideBy;
    }
}

size_t NetworkWeightGradients::numLayers() const
{
    return weightGradients.size();
}

void NetworkWeightGradients::setToZero() {
    for (LayerWeightsT& lWeights: weightGradients ) {
        lWeights.setZero();
    }
}

//***********//

NetworkLayerGradients::NetworkLayerGradients(NNetwork& network)
{
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        layerGradients.push_back(network.layer(layerPos).getBiases());
        layerGradients[layerPos].setZero();
    }
}

void NetworkLayerGradients::setLayerGradients(const SingleRowT& newLayerGrads, size_t layer)
{
    if(layer >= layerGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    layerGradients[layer] = (newLayerGrads);
}

const SingleRowT& NetworkLayerGradients::getLayerGradients(size_t layer) const
{
    if(layer >= layerGradients.size())
    {
        throw std::out_of_range("Layer does not exist");
    }
    return layerGradients[layer];
}

size_t NetworkLayerGradients::numLayers() const
{
    return layerGradients.size();
}

void NetworkLayerGradients::setToZero() {
    for(SingleRowT& layer: layerGradients) {
        layer.setZero();
    }
}

void NetworkLayerGradients::numericAddLayerGradients(const NetworkLayerGradients& layerGradsToAdd)
{
    // this function performs a numeric add, taking layerGradsToAdd and adding to the NetWorkLayerGradients
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

void NetworkLayerGradients::divideLayerGradients(size_t divideBy)
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
    double correct = 0;
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
            // accurate if highest probability prediction matches the answer
            if(item.labels.coeff(0, posOfHighestElement) == 1)
            {
                correct++;
            }
        }
        // how is accuracy calculated for mse? not sure it makes sense and how it would differ from loss

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
    // this function calculates the derivative of the output of a layer wrt to the net input (the derivative thus depends upon the activation function)

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

SingleRowT calculateOutputLayerGradientsForExampleItem(const NLayer& outputLayer, ActFunc actFuncForOutputLayer, LossFunc lossFunc, const Labels& targets)
{
    // This function calculates the derivative of the error wrt to the net input to the final layer

    if (lossFunc == LossFunc::CROSS_ENTROPY)
    {
        // simplified calculation of derivative for cross entropy loss and softmax activation (this is just the actual output - ground truth)
        return outputLayer.getOutputs() - targets;
    }
    if (lossFunc == LossFunc::MSE)
    {
        // first calculate the derivative of the error wrt to the output of the final layer
        SingleRowT gradientOfMse = outputLayer.getOutputs() - targets;
        // then calculate the derivative of the output of the final layer wrt to the net input (i.e the derivative of the activation function)
        SingleRowT gradientOfActFunc = calculateActivationFunctionGradients(outputLayer, actFuncForOutputLayer);
        // then calculate the derivative of the error wrt to the net input
        return gradientOfMse.array() * gradientOfActFunc.array();
    }
    else {
        throw std::runtime_error("Unsupported loss function.");
    }
}

void calculateHiddenLayerGradientsForExampleItem(NNetwork& network, const ActFuncList& actFuncs, NetworkLayerGradients& layerGrads)
{
    const size_t lastHiddenLayer = network.numLayers() - 2; // -1 is the output layer so -2 is last hidden layer
    // reverse backwards through each layer
    for (size_t layerPos = lastHiddenLayer; layerPos != (size_t) - 1;--layerPos)
    {
        const LayerWeightsT& weightsOfSubsequentLayer = network.layer(layerPos + 1).getWeights();
        const SingleRowT& subsequentLayerGrads = layerGrads.getLayerGradients(layerPos + 1);
        //
        SingleRowT errorWrtOutput = subsequentLayerGrads * weightsOfSubsequentLayer.transpose();
        // calculate the derivative of the output of the layer wrt to the net input
        SingleRowT activationFunctionGradient = calculateActivationFunctionGradients(network.layer(layerPos), actFuncs[layerPos]);
        // calculate the derivative of the error wrt to the net input
        SingleRowT errorWrtNetInput = errorWrtOutput.array() * activationFunctionGradient.array();

        if (!errorWrtNetInput.allFinite())
        {
            throw std::logic_error("(1) Contains INF or NaN");
        }
        layerGrads.setLayerGradients(errorWrtNetInput, layerPos);
    }
}

void calculateWeightGradientsForExampleItem(NNetwork& network, const NetworkLayerGradients& layerGrads, NetworkWeightGradients& weightGrads)
{
    // move from the weights for the output layer back through the weights for hidden layers of the network
    for(size_t layerPos = network.numLayers() - 1; layerPos != (size_t) - 1 ; --layerPos)
    {
        SingleRowT prevLayerOutput;
        if(layerPos > 0)
        {
            prevLayerOutput = network.layer(layerPos - 1).getOutputs();
        }
        else
        {
            prevLayerOutput = network.getInputs(); // if layer is first hidden layer (layer 0) then output of previous layer is input
        }
        const SingleRowT& currentLayerGrad = layerGrads.getLayerGradients(layerPos);
        // the gradients of weights can be calculated as the matrix multiplication of the transpose of the output of the  layer preceding the weights
        // multiplied by the gradients of the layer succeeding the weights (already calculated).
        LayerWeightsT weightGradsForLayer = prevLayerOutput.transpose() * currentLayerGrad;
        weightGrads.setWeightGradientsForLayer(weightGradsForLayer, layerPos);
    }
}

void calculateGradientsForExampleItem(NNetwork& network, const ActFuncList& actFuncs, LossFunc lossFunc, const ExampleItem& trItem, NetworkLayerGradients& layerGrads, NetworkWeightGradients& weightGrads, NetNumT dropOutRate)
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

    // calculate the FINAL LAYER gradients
    const size_t outputLayerPos = network.numLayers() - 1;
    const SingleRowT outputLayerGradients = calculateOutputLayerGradientsForExampleItem(network.outputLayer(), actFuncs[outputLayerPos], lossFunc, trItem.labels);
    if(!outputLayerGradients.allFinite())
    {
        throw std::logic_error("(3) Contains INF or NaN");
    }
    layerGrads.setLayerGradients(outputLayerGradients, outputLayerPos);

    // calculate the HIDDEN LAYER gradients
    calculateHiddenLayerGradientsForExampleItem(network, actFuncs,layerGrads);

    // calculate the WEIGHT gradients
    calculateWeightGradientsForExampleItem(network, layerGrads, weightGrads);
}

void updateNetworkUsingGradients(NNetwork& network, const NetworkLayerGradients& layerGrads, const NetworkWeightGradients& weightGrads, const LearningRateList& learningRatesPerLayer, NetNumT momentumFactor, NetworkLayerGradients& prevUpdateBiasDelta, NetworkWeightGradients& prevUpdateWeightDelta)
{
    if(learningRatesPerLayer.size() != network.numLayers())
    {
        throw std::logic_error("Number of learning rate layers does not match number of network layers");
    }

    //  go backwards through the network starting at the output layer
    size_t layerPos = network.numLayers() - 1;
    do
    {
        // each layer has its own learning rate
        const NetNumT learningRateForLayer = learningRatesPerLayer[layerPos];

        // update weights

        const LayerWeightsT& layerWeights = network.layer(layerPos).getWeights(); // the weights for this layer
        const LayerWeightsT& layerWeightGrads = weightGrads.getWeightGradientsForLayer(layerPos); // the gradients for the weights of this layer
        const LayerWeightsT& layerWeightsDelta = ((1.0 - momentumFactor) * layerWeightGrads) + // momentum based calculation
                                                    (momentumFactor * prevUpdateWeightDelta.getWeightGradientsForLayer(layerPos));
        network.layer(layerPos).setWeights(layerWeights - (learningRateForLayer * layerWeightsDelta)); // update by learning rate * gradients
        prevUpdateWeightDelta.setWeightGradientsForLayer(layerWeightsDelta, layerPos);

        // update biases

        const SingleRowT& layerBiases = network.layer(layerPos).getBiases();
        const SingleRowT& layerBiasGrads = layerGrads.getLayerGradients(layerPos);
        const SingleRowT& layerBiasDelta = ((1 - momentumFactor) * layerBiasGrads) + (momentumFactor * prevUpdateBiasDelta.getLayerGradients(layerPos));
        network.layer(layerPos).setBiases(layerBiases - (learningRateForLayer * layerBiasDelta));
        prevUpdateBiasDelta.setLayerGradients(layerBiasDelta, layerPos);
    } while (layerPos-- != 0);
}

void calculateGradientsOverBatch(NNetwork& network, ExampleData::iterator batchStart, ExampleData::iterator batchEnd, const ActFuncList& actFuncs, LossFunc lossFunc, NetworkLayerGradients& averagedLayerGrads, NetworkWeightGradients& averagedWeightGrads, NetNumT dropOutRate)
{
    // these are the gradients for each item in the batch (used to calculate the average gradients passed as a parameter to this method)
    NetworkLayerGradients layerGradientsForItem(network);
    NetworkWeightGradients weightGradientsForItem(network);

    for(auto trItemIt = batchStart; trItemIt != batchEnd; ++trItemIt)
    {
        // calculate gradients for the item in the minibatch
        calculateGradientsForExampleItem(network, actFuncs, lossFunc, *trItemIt, layerGradientsForItem, weightGradientsForItem, dropOutRate);
        // add calculated gradients for item to running total
        averagedLayerGrads.numericAddLayerGradients(layerGradientsForItem);
        averagedWeightGrads.numericAddWeightGradients(weightGradientsForItem);
    }
    // divide running total of gradients to find average
    averagedLayerGrads.divideLayerGradients(std::distance(batchStart, batchEnd));
    averagedWeightGrads.divideWeightGradients(std::distance(batchStart, batchEnd));
}

void train(NNetwork& network, ExampleData& trainingData, const ActFuncList& actFuncs, LossFunc lossFunc, const LearningRateList& lrList, NetNumT momentum, InitMethod initMethod, size_t epochsToRun, size_t batchSz, const ExampleData& testData, NetNumT dropOutRate)
{
    if (!isTrainingDataValid(network.classes(), trainingData, network.getInputs().size()))
    {
        throw std::logic_error("Training data invalid");
    }
    initialiseWeightsBiases(network, initMethod);

    // prev weight updates for momentum - set to 0 for first update
    NetworkWeightGradients prevWeightDelta(network);
    NetworkLayerGradients prevBiasDelta(network);

    // these contain the gradients for each (mini) batch - declared here to save time from reinitialising in each loop
    NetworkLayerGradients lGradsOverBatch(network);
    NetworkWeightGradients wGradsOverBatch(network);

    for(size_t epoch = 0; epoch < epochsToRun; ++epoch)
    {
        auto start = std::chrono::steady_clock::now();
        // random shuffle and then update for each minibatch
        std::shuffle(trainingData.begin(), trainingData.end(), std::default_random_engine(12345));
        // loop through the training data in the batch size
        for(auto trItemIt = trainingData.begin(); trItemIt < trainingData.end(); trItemIt += static_cast<std::vector<ExampleItem>::difference_type>(batchSz))
        {
            auto batchEnd = trItemIt + static_cast<std::vector<ExampleItem>::difference_type>(batchSz);
            if(batchEnd > trainingData.end())
            {
                batchEnd = trainingData.end();
            }
            // calculate the average gradients over the batch
            calculateGradientsOverBatch(network, trItemIt, batchEnd, actFuncs, lossFunc, lGradsOverBatch, wGradsOverBatch, dropOutRate);
            // update the network with the averaged gradients
            updateNetworkUsingGradients(network, lGradsOverBatch, wGradsOverBatch, lrList, momentum, prevBiasDelta, prevWeightDelta);
            // clear averaged  gradients - is this necessary?
            wGradsOverBatch.setToZero();
            lGradsOverBatch.setToZero();
        }

        auto end = std::chrono::steady_clock::now();

        // Print

        std::cout << "Threads: " << Eigen::nbThreads() << std::endl;
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