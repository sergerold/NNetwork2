//
// Created by Lenovo on 10/04/2023.
//

#ifndef NNETWORK2_NLAYER_H
#define NNETWORK2_NLAYER_H

#include "Eigen/Dense"

using NetNumT = double;
using SingleRowT = Eigen::Matrix<NetNumT, 1, Eigen::Dynamic>;
//using SingleRowT = Eigen::Vector<NetNumT , Eigen::Dynamic>;


class NLayer
{
    private:
        SingleRowT mLayerBiases;
        SingleRowT mLayerOutputs;
        Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic> mLayerWeights;
        
        void resizeLayer(size_t newLayerSz);
        void resizeNumWeightsPerNeuron(size_t newWeightsSz);

    public:
        explicit NLayer(size_t layerSz, size_t numIncomingWeightsToEachNeuron);

        SingleRowT& getBiases();
        const SingleRowT& getOutputs();

        Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic>& getWeights();
        Eigen::Ref<Eigen::VectorXd> getWeightsForNeuron(size_t neuronPos);

        size_t size();

        friend class NNetwork;
};

#endif //NNETWORK2_NLAYER_H
