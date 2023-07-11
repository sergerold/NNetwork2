//
// Created by Lenovo on 08/05/2023.
//

#ifndef NNETWORK2_TRAINING_H
#define NNETWORK2_TRAINING_H

#include "NNetwork.h"

#include <vector>

// types

using Labels = std::map<ClassT, NetNumT>;

struct TrainingItem
{
    InputList inputs;
    Labels labels;
};
using TrainingData = std::vector<TrainingItem>;

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
        void insertWeightGradientsForLayer(const LayerWeightsT& newWeightGrads, size_t layer);
        const LayerWeightsT& getWeightGradientsForLayer(size_t layer) const;
        void addWeightGradients(const WeightGradients& weightsToAdd);
        void divideWeightGradients(size_t divideBy);
        size_t numLayers() const;
};

class LayerGradients
{
    private:
        std::vector<SingleRowT> layerGradients;

    public:
        explicit LayerGradients(size_t numLayers);
        void insertLayerGradients(const SingleRowT& newLayerGrads, size_t layer);
        const SingleRowT& getLayerGradients(size_t layer) const;
        void addLayerGradients(const LayerGradients& layerGradsToAdd);
        void divideLayerGradients(size_t divideBy);
        size_t numLayers() const;
};

using LearningRateList = std::vector<NetNumT>;

// TRAINING ALGORITHMS

// data initialisation functions
enum class InitMethod
{
        RANDOM,
        HE,
};
void initialiseWeightsBiases(NNetwork& network, InitMethod method, const ActFuncList& actFuncs);

// loss functions
NetNumT calculateLossForTrainingItem(const Labels& labels, LossFunc lossFunc, const SingleRowT& networkOut);
NetNumT calculateLossForTrainingData(NNetwork& network, const TrainingData& trData, const ActFuncList& actFuncs, LossFunc lossFunc);

// Gradient calculation

SingleRowT calculateActivationFunctionGradients(const NLayer& layer, ActFunc actFunc);

SingleRowT calculateOutputLayerGradientsForTrainingItem(const NLayer& outputLayer, ActFunc actFuncForLayer, LossFunc lossFunc, const Labels& targets);
void calculateHiddenLayerGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LayerGradients& layerGrads);
void calculateWeightGradientsForTrainingItem(NNetwork& network, const LayerGradients& layerGrads, WeightGradients& weightGrads);

void calculateGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LossFunc lossFunc, TrainingItem& trItem, LayerGradients& layerGrads, WeightGradients& weightGrads);
void calculateGradientsOverBatch(NNetwork& network, TrainingData& trData, TrainingData::iterator batchStart, size_t batchSz, const ActFuncList& actFuncs, LossFunc lossFunc, LayerGradients& layerGrads, WeightGradients& weightGrads);

// TRAIN
void updateNetworkWeightsBiasesWithGradients(NNetwork& network, const LayerGradients& layerGrads, const WeightGradients& weightGrads, const LearningRateList& learningRatesPerLayer);

void train(NNetwork& network, TrainingData trData, ActFuncList actFuncs, LossFunc lossFunc, LearningRateList lrList, InitMethod initMethod, size_t epochsToRun, size_t batchSz);

#endif //NNETWORK2_TRAINING_H