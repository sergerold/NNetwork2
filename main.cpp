#include <iostream>
#include "Eigen/Dense"

#include "NNetwork.h"
#include "Training.h"

using Eigen::MatrixXd;

int main()
{
    std::function<NetNumT (NetNumT)> sigmoid = [](NetNumT input){return 1/(1 + exp( -input));};



    Eigen::Matrix<NetNumT, 1, Eigen::Dynamic> inputs;
    inputs.resize(1, 2);
    inputs << 0.6, 0.3;
    LabelList labels {"A", "B"};

    NNetwork network(inputs, labels);

    network.addLayer(2, 0);


    network.layer(0).getBiases() << 0.5, 0.8;
    network.layer(1).getBiases() << 0.2, 0.4;

    network.layer(0).getWeights() << 0.22, 0.44, 0.66, 0.88;
    network.layer(1).getWeights() << 0.33, 0.55, 0.77, 0.99;

    network.feedforward(sigmoid);
    std::cout << network.getOutput("A") << " , " << network.getOutput("B") << std::endl;

    //TrainingItem trainingItem;
    //trainingItem.inputs = SingleRowT {{0.6, 0.3}};
    //trainingItem.targets.emplace("A", 0.737);
    //trainingItem.targets.emplace("B", 0.807);



}