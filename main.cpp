#include <iostream>
#include "Eigen/Dense"

#include "NNetwork.h"
#include "Training.h"
#include "Debug.h"


int main()
{
    Eigen::Matrix<NetNumT, 1, Eigen::Dynamic> inputs;
    inputs.resize(1, 2);
    inputs << 0.05, 0.1;
    ClassList classes {"O1", "O2"};
    NNetwork network(inputs, classes);
    network.addLayer(2, 0);

    Eigen::Matrix<NetNumT, 1, 2> l0B, l1B;
    l0B << 0.35, 0.35;
    l1B << 0.6, 0.6;

    network.layer(0).setBiases( l0B ) ;
    network.layer(1).setBiases( l1B ) ;

    Eigen::Matrix<NetNumT, 2, 2> l0W, l1W;
    l0W << 0.15, 0.25, 0.20, 0.30;
    l1W << 0.40, 0.50, 0.45, 0.55;

    network.layer(0).setWeights(l0W);
    network.layer(1).setWeights(l1W);


    TrainingItem trItem;
    trItem.labels.emplace("O1", 0.01);
    trItem.labels.emplace("O2", 0.99);
    trItem.inputs = inputs;

    LearningRateList lRList = {0.5, 0.5};
    ActFuncList actFuncs = ActFuncList{ActFunc::RELU, ActFunc::RELU};
    LossFunc lossFunc = LossFunc::MSE;

    for(size_t i = 0; i < 0; ++i)
    {
        LayerGradients lGrads(network.numLayers());
        WeightGradients wGrads(network.numLayers());

        calculateGradientsForTrainingItem(network, actFuncs, lossFunc, trItem,lGrads, wGrads) ;
        updateNetworkWeightsBiasesWithGradients(network, lGrads,wGrads, lRList);

        std::cout << "Error: " << calculateLoss(trItem.labels, lossFunc, network.outputLayer().getOutputs()) << std::endl;
    }

    initialiseWeightsBiases(network, InitMethod::RANDOM, actFuncs);
    printNetwork(network);
}