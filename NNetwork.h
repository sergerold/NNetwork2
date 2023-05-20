//
// Created by Lenovo on 10/04/2023.
//

#ifndef NNETWORK2_NNETWORK_H
#define NNETWORK2_NNETWORK_H

#include "NLayer.h"

#include <vector>
#include <functional>
#include <map>
#include <set>
#include "Eigen/Dense"

using LabelT = std::string;
using LabelList = std::set<LabelT>;
using InputList = SingleRowT;

class NNetwork
{
    private:
        std::vector<NLayer> mNLayer;
        std::map<LabelT, size_t> mOutputLabels;

        const size_t INPUT_LAYER_OFFSET = 1;

    public:
        NNetwork(const SingleRowT& inputs, const LabelList& labels);

        NLayer& layer(size_t layer);
        NLayer& outputLayer();
        size_t numLayers();
        NetNumT getOutput(const LabelT&);

        const SingleRowT& getInputs();
        void setInputs(const SingleRowT& inputs);

        bool addLayer(size_t layerSz, size_t insertPos);
        bool changeLayerSz(size_t layer, size_t newLayerSz);

        void feedforward(std::function<NetNumT (NetNumT)>& actFunc);
};

#endif //NNETWORK2_NNETWORK_H