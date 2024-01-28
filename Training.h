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

class WeightGradients
{
    private:
        std::vector< LayerWeightsT > weightGradients;

    public:
        explicit WeightGradients(size_t numLayers);
        explicit WeightGradients(NNetwork& network);

        void insertWeightGradientsForLayer(const LayerWeightsT& newWeightGrads, size_t layer);
        [[nodiscard]] const LayerWeightsT& getWeightGradientsForLayer(size_t layer) const;
        void addWeightGradients(const WeightGradients& weightsToAdd);
        void divideWeightGradients(size_t divideBy);
        [[nodiscard]] size_t numLayers() const;
};

class LayerGradients
{
    private:
        std::vector<SingleRowT> layerGradients;

    public:
        explicit LayerGradients(size_t numLayers);
        explicit LayerGradients(NNetwork& network);

        void insertLayerGradients(const SingleRowT& newLayerGrads, size_t layer);
        [[nodiscard]] const SingleRowT& getLayerGradients(size_t layer) const;
        void addLayerGradients(const LayerGradients& layerGradsToAdd);
        void divideLayerGradients(size_t divideBy);
        [[nodiscard]] size_t numLayers() const;
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

SingleRowT calculateOutputLayerGradientsForTrainingItem(const NLayer& outputLayer, ActFunc actFuncForLayer, LossFunc lossFunc, const Labels& targets);
void calculateHiddenLayerGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LayerGradients& layerGrads);
void calculateWeightGradientsForTrainingItem(NNetwork& network, const LayerGradients& layerGrads, WeightGradients& weightGrads);

void calculateGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LossFunc lossFunc, const ExampleItem& trItem, LayerGradients& layerGrads, WeightGradients& weightGrads, NetNumT dropOutRate);
void calculateGradientsOverBatch(NNetwork& network, ExampleData::iterator batchStart, ExampleData::iterator batchEnd, const ActFuncList& actFuncs, LossFunc lossFunc, LayerGradients& layerGrads, WeightGradients& weightGrads, NetNumT dropOutRate);

// TRAIN
void updateNetworkWeightsBiasesWithGradients(NNetwork& network, const LayerGradients& layerGrads, const WeightGradients& weightGrads, const LearningRateList& learningRatesPerLayer, NetNumT momentumFactor, LayerGradients& prevUpdateBiasDelta, WeightGradients& prevUpdateWeightDelta);
void train(NNetwork& network, ExampleData& trainingData, const ActFuncList& actFuncs, LossFunc lossFunc, const LearningRateList& lrList, NetNumT momentum, InitMethod initMethod, size_t epochsToRun, size_t batchSz, const ExampleData& testData, NetNumT dropOutRate);

#endif //NNETWORK2_TRAINING_H