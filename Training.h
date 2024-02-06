//
// Created by Lenovo on 08/05/2023.
//

#ifndef NNETWORK2_TRAINING_H
#define NNETWORK2_TRAINING_H

#include "NNetwork.h"

#include <vector>

// types

using Labels = SingleRowT;

struct ExampleItem
{
    InputList inputs;
    Labels labels;
};
using ExampleData = std::vector<ExampleItem>;

enum class LossFunc
{
        MSE,
        CROSS_ENTROPY
};

// gradients for each weight in the network
class NetworkWeightGradients
{
    private:
        std::vector< LayerWeightsT > weightGradients;

    public:
        explicit NetworkWeightGradients(NNetwork& network);

        void setWeightGradientsForLayer(const LayerWeightsT& newWeightGrads, size_t layer);
        [[nodiscard]] const LayerWeightsT& getWeightGradientsForLayer(size_t layer) const;
        void numericAddWeightGradients(const NetworkWeightGradients& weightsToAdd);
        void divideWeightGradients(size_t divideBy);
        [[nodiscard]] size_t numLayers() const;
        void setToZero();
};


// Gradients for each neuron in the network (i.e. gradients for the bias)
class NetworkLayerGradients
{
    private:
        std::vector<SingleRowT> layerGradients;

    public:
        explicit NetworkLayerGradients(NNetwork& network);

        void setLayerGradients(const SingleRowT& newLayerGrads, size_t layer);
        [[nodiscard]] const SingleRowT& getLayerGradients(size_t layer) const;
        void numericAddLayerGradients(const NetworkLayerGradients& layerGradsToAdd);
        void divideLayerGradients(size_t divideBy);
        [[nodiscard]] size_t numLayers() const;
        void setToZero();
};

using LearningRateList = std::vector<NetNumT>;

// TRAINING ALGORITHMS

// wieght initialisation functions
enum class InitMethod
{
        RANDOM_UNIFORM,
        NORMALISED_HE, // Use for ReLU
        UNIFORM_HE,
        NORMALISED_XAVIER, // Use for Sigmoid or Tan Act funcs
        UNIFORM_XAVIER,
        NO_INIT
};
void initialiseWeightsBiases(NNetwork& network, InitMethod method);

// loss functions / accuracy calculations
NetNumT calculateLossForExampleItem(const Labels& labels, LossFunc lossFunc, const SingleRowT& networkOut);
NetNumT calculateLossForExampleData(NNetwork& network, const ExampleData& trData, const ActFuncList& actFuncs, LossFunc lossFunc);

NetNumT calculateAccuracyForExampleData(NNetwork& network, const ExampleData& data, const ActFuncList &actFuncList);

// Gradient calculation

SingleRowT calculateActivationFunctionGradients(const NLayer& layer, ActFunc actFunc);

SingleRowT calculateOutputLayerGradientsForExampleItem(const NLayer& outputLayer, ActFunc actFuncForOutputLayer, LossFunc lossFunc, const Labels& targets);
void calculateHiddenLayerGradientsForExampleItem(NNetwork& network, const ActFuncList& actFuncs, NetworkLayerGradients& layerGrads);
void calculateWeightGradientsForExampleItem(NNetwork& network, const NetworkLayerGradients& layerGrads, NetworkWeightGradients& weightGrads);

void calculateGradientsForExampleItem(NNetwork& network, const ActFuncList& actFuncs, LossFunc lossFunc, const ExampleItem& trItem, NetworkLayerGradients& layerGrads, NetworkWeightGradients& weightGrads, NetNumT dropOutRate);
void calculateGradientsOverBatch(NNetwork& network, ExampleData::iterator batchStart, ExampleData::iterator batchEnd, const ActFuncList& actFuncs, LossFunc lossFunc, NetworkLayerGradients& averagedLayerGrads, NetworkWeightGradients& averagedWeightGrads, NetNumT dropOutRate);

// TRAIN
void updateNetworkUsingGradients(NNetwork& network, const NetworkLayerGradients& layerGrads, const NetworkWeightGradients& weightGrads, const LearningRateList& learningRatesPerLayer, NetNumT momentumFactor, NetworkLayerGradients& prevUpdateBiasDelta, NetworkWeightGradients& prevUpdateWeightDelta);
void train(NNetwork& network, ExampleData& trainingData, const ActFuncList& actFuncs, LossFunc lossFunc, const LearningRateList& lrList, NetNumT momentum, InitMethod initMethod, size_t epochsToRun, size_t batchSz, const ExampleData& testData, NetNumT dropOutRate);

#endif //NNETWORK2_TRAINING_H