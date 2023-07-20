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

#include "DataSpecs.h"

using ClassT = std::string;
using ClassList = std::set<ClassT>;
using InputList = Eigen::Matrix<INPUT_TYPE, 1, Eigen::Dynamic>;

enum class ActFunc
{
        SIGMOID = 0,
        RELU = 1,
        SOFTMAX = 2
};

using ActFuncList = std::vector<ActFunc>;

class NNetwork
{
    private:
        std::vector<NLayer> mNLayer;
        std::map<ClassT, size_t> mOutputClasses; // ordered list of classes

        const size_t INPUT_LAYER_OFFSET = 1;

        void applyActFuncToLayer(SingleRowT& netInputs, ActFunc actFunc);

    public:
        NNetwork(size_t inputSz, const ClassList& labels);

        NLayer& layer(size_t layer);
        NLayer& outputLayer();
        size_t numLayers() const;
        NetNumT getOutput(const ClassT&) const;

        const SingleRowT& getInputs() const;
        void setInputs(const SingleRowT& inputs);

        bool addLayer(size_t layerSz, size_t insertPos);
        bool changeLayerSz(size_t layer, size_t newLayerSz);

        const std::map<ClassT, size_t>& classes() const;

        void feedforward(const ActFuncList& actFuncs);

};


#endif //NNETWORK2_NNETWORK_H