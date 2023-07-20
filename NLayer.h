//
// Created by Lenovo on 10/04/2023.
//

#ifndef NNETWORK2_NLAYER_H
#define NNETWORK2_NLAYER_H

#include "Eigen/Dense"

using NetNumT = float;
using LayerWeightsT =  Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic>;
using SingleRowT = Eigen::Matrix<NetNumT, 1, Eigen::Dynamic>;

class NLayer
{
    private:
        SingleRowT mLayerBiases;
        SingleRowT mLayerOutputs;
        LayerWeightsT mLayerWeights;
        
        void resizeLayer(size_t newLayerSz);
        void resizeNumWeightsPerNeuron(size_t newWeightsSz);

    public:
        explicit NLayer(size_t layerSz, size_t numIncomingWeightsToEachNeuron);

        const SingleRowT& getBiases();
        void setBiases(const SingleRowT& biases);

        const SingleRowT& getOutputs() const;

        const LayerWeightsT& getWeights();
        void setWeights(const LayerWeightsT& weights);

        size_t size();

        friend class NNetwork;
};

#endif //NNETWORK2_NLAYER_H
