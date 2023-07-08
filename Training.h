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
        void addWeightGradientsForLayer(const LayerWeightsT& newWeightGrads, size_t layer);
        const LayerWeightsT& getWeightGradientsForLayer(size_t layer) const;
        size_t numLayers();
};

class LayerGradients
{
    private:
        std::vector<SingleRowT> layerGradients;

    public:
        explicit LayerGradients(size_t numLayers);
        void addLayerGradients(const SingleRowT& newLayerGrads, size_t layer);
        const SingleRowT& getLayerGradients(size_t layer) const;
        size_t numLayers();
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

// data funcs
SingleRowT trainingItemToVector(const std::map<ClassT, NetNumT>& trItem);
bool isTrainingDataValid(const std::map<ClassT, size_t>& networkLabels, const TrainingData& trainingData, size_t networkInputSz);

// loss functions
NetNumT calculateLoss(const Labels& labels, LossFunc lossFunc, const SingleRowT& networkOut);

// Gradient calculation

SingleRowT calculateActivationFunctionGradients(const NLayer& layer, ActFunc actFunc);

void calculateGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LossFunc lossFunc, TrainingItem& trItem, LayerGradients& layerGrads, WeightGradients& weightGrads);

SingleRowT calculateOutputLayerGradientsForTrainingItem(const NLayer& outputLayer, ActFunc actFuncForLayer, LossFunc lossFunc, const Labels& targets);
void calculateHiddenLayerGradientsForTrainingItem(NNetwork& network, const ActFuncList& actFuncs, LayerGradients& layerGrads);
void calculateWeightGradientsForTrainingItem(NNetwork& network, const LayerGradients& layerGrads, WeightGradients& weightGrads);

void updateNetworkWeightsBiasesWithGradients(NNetwork& network, const LayerGradients& layerGrads, const WeightGradients& weightGrads, const LearningRateList& learningRatesPerLayer);

#endif //NNETWORK2_TRAINING_H