//
// Created by Lenovo on 09/07/2023.
//
#include <fstream>

#include <iostream>
#include <set>
#include "Data.h"

bool isTrainingDataValid(const std::map<ClassT, size_t>& networkLabels, const TrainingData& trainingData, size_t networkInputSz)
{
    for(const auto& item : trainingData)
    {
        const Labels& targets = item.labels;
        if (targets.size() != networkLabels.size())
        {
            return false;
        }
        for (const auto & networkLabel : networkLabels)
        {
            if (targets.count(networkLabel.first) < 1)
            {
                return false;
            }
        }
        if (item.inputs.size() != networkInputSz)
        {
            return false;
        }
    }
    return true;
}

SingleRowT trainingItemToVector(const std::map<ClassT, NetNumT>& trItem)
{
    size_t count = 0;
    SingleRowT targetValuesAsVector;
    targetValuesAsVector.resize(Eigen::Index(trItem.size()));
    for(auto & it : trItem)
    {
        targetValuesAsVector(0, Eigen::Index (count) ) = it.second;
        count++;
    }
    return targetValuesAsVector;
}

void normaliseTrainingData(TrainingData& trData, NormalisationMethod method)
{
    // convert inputs in trData to matrix - every col is a list of an input element over every training item
    Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic> inputsAsMatrix;
    inputsAsMatrix.resize(trData.size(), trData[0].inputs.size());
    for(size_t pos = 0; pos < trData.size(); ++pos)
    {
        inputsAsMatrix.row(pos) = trData[pos].inputs;
    }
    // normalise
    if(method == NormalisationMethod::Z_SCORE)
    {
        // iterate over each set of inputElements in trData
        for(size_t inputElement = 0; inputElement < trData[0].inputs.size(); ++inputElement)
        {
            auto inputElementAcrossItems = inputsAsMatrix.col(inputElement);
            NetNumT mean = inputElementAcrossItems.array().mean();
            NetNumT sd = sqrt ( (inputElementAcrossItems.array() - mean).pow(2).sum() /
                                NetNumT (inputElementAcrossItems.size()) );

            inputElementAcrossItems = (inputElementAcrossItems.array() - mean) / sd;
            inputsAsMatrix.col(inputElement) = inputElementAcrossItems;
        }
    }
    if(method == NormalisationMethod::MINMAX)
    {
        for(size_t inputElement = 0; inputElement < trData[0].inputs.size(); ++inputElement) {

            auto inputElementAcrossItems = inputsAsMatrix.col(inputElement);
            NetNumT maxInputValue = inputElementAcrossItems.array().maxCoeff();
            NetNumT minInputValue = inputElementAcrossItems.array().minCoeff();
            inputElementAcrossItems = (inputElementAcrossItems.array() - minInputValue) / (maxInputValue - minInputValue);
            inputsAsMatrix.col(inputElement) = inputElementAcrossItems;
        }
    }
    if(method == NormalisationMethod::LOG)
    {
        inputsAsMatrix = inputsAsMatrix.array().log();
    }
    // put data back in trData
    for(size_t trItemPos = 0; trItemPos < trData.size(); ++trItemPos)
    {
        trData[trItemPos].inputs = inputsAsMatrix.row(trItemPos);
    }
}

std::set<std::string> getClasses()
{
    return ClassList {"versicolor", "setosa", "virginica"};
}

NetNumT getInputSz()
{
    return 4;
}

TrainingData loadTrainingDataFromFile(std::string fName)
{
    TrainingData trData;
    std::ifstream dataFile;
    dataFile.open(fName, std::ifstream::in);

    if (! dataFile.is_open())
    {
        throw std::logic_error("Could not open file");
    }

    std::string lineOfFile, delimiter = ",";
    while (std::getline(dataFile, lineOfFile)) {
        lineOfFile += delimiter; // add delimiter to end of line so as to easily parse
        TrainingItem trItem;
        trItem.inputs.resize(1, getInputSz());

        // load expected outcome
        size_t pos = lineOfFile.find(delimiter);
        std::string targetNum = lineOfFile.substr(0, pos);
        lineOfFile.erase(0, pos + delimiter.length());

        ClassList classes = getClasses();

        for(auto it = classes.begin(); it != classes.end(); ++it)
        {
            trItem.labels[*it] = 0;
        }
        trItem.labels[targetNum] = 1;

        // add inputs
        std::string inputAsStr;
        size_t inputCount = 0;

        while ((pos = lineOfFile.find(delimiter)) != std::string::npos)
        {
            inputAsStr = lineOfFile.substr(0, pos);
            lineOfFile.erase(0, pos + delimiter.length());
            NetNumT inputAsNum = std::stod(inputAsStr);
            trItem.inputs.row(0).col(inputCount) << inputAsNum;
            inputCount++;
        }
        trData.push_back(trItem);

    }
    dataFile.close();
    return trData;
}