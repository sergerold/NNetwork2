//
// Created by Lenovo on 09/07/2023.
//
#include <fstream>
#include <iostream>
#include <set>

#include "Data.h"
#include "DataSpecs.h"

bool isTrainingDataValid(const std::map<ClassT, size_t>& networkLabels, const ExampleData& trainingData, size_t networkInputSz)
{
    for(const auto& item : trainingData)
    {
        const Labels& targets = item.labels;
        if (targets.size() != networkLabels.size())
        {
            std::cout << targets.size() << ", " << networkLabels.size() << std::endl;
            return false;
        }
        if (item.inputs.size() != networkInputSz)
        {
            std::cout << "FAIL 2" << std::endl;
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

bool areInputElementsDifferent(const Eigen::Matrix<INPUT_TYPE, Eigen::Dynamic, Eigen::Dynamic>& inputElements)
{
    NetNumT firstInput = inputElements.coeff(0, 0);
    for(Eigen::Index row = 0; row < inputElements.rows(); ++row)
    {
        for(Eigen::Index col = 0; col < inputElements.cols(); ++col)
        {
            if(inputElements.coeff(row, col) != firstInput)
            {
                return true;
            }
        }
    }
    return false;
}

void normaliseTrainingData(ExampleData& trData, DataNormalisationMethod method)
{
    // convert inputs in trData to matrix - every col is a list of an input element over every training item
    Eigen::Matrix<INPUT_TYPE, Eigen::Dynamic, Eigen::Dynamic> inputsAsMatrix;
    inputsAsMatrix.resize(Eigen::Index(trData.size()), Eigen::Index(trData[0].inputs.size()));
    for(size_t pos = 0; pos < trData.size(); ++pos)
    {
        inputsAsMatrix.row(Eigen::Index(pos)) = trData[pos].inputs;
    }
    // normalise
    if(method == DataNormalisationMethod::Z_SCORE)
    {
        // iterate over each set of inputElements in trData
        for(size_t inputElement = 0; inputElement < trData[0].inputs.size(); ++inputElement)
        {
            Eigen::Matrix<INPUT_TYPE, 1, Eigen::Dynamic> inputElementAcrossItems = inputsAsMatrix.col(Eigen::Index(inputElement));
            // no valid z score if all elements same so do not amend
            if(!areInputElementsDifferent(inputElementAcrossItems))
            {
                continue;
            }
            NetNumT mean = inputElementAcrossItems.array().mean();
            NetNumT sd = sqrt ( (inputElementAcrossItems.array() - mean).pow(2).sum() /
                                NetNumT (inputElementAcrossItems.size()) );

            inputElementAcrossItems = (inputElementAcrossItems.array() - mean) / sd;
            inputsAsMatrix.col(Eigen::Index(inputElement)) = inputElementAcrossItems;
        }
    }
    if(method == DataNormalisationMethod::MINMAX)
    {
        for(size_t inputElement = 0; inputElement < trData[0].inputs.size(); ++inputElement)
        {
            auto inputElementAcrossItems = inputsAsMatrix.col(Eigen::Index(inputElement));
            NetNumT maxInputValue = inputElementAcrossItems.array().maxCoeff();
            NetNumT minInputValue = inputElementAcrossItems.array().minCoeff();

            // DO NOT NORMALISE IF VALUES ALL SAME (DIVIDING BY 0 GIVES ERROR)
            if(minInputValue == maxInputValue)
            {
                continue;
            }

            inputElementAcrossItems = (inputElementAcrossItems.array() - minInputValue) / (maxInputValue - minInputValue);
            inputsAsMatrix.col(Eigen::Index(inputElement)) = inputElementAcrossItems;
        }
    }
    if(method == DataNormalisationMethod::LOG)
    {
        for(Eigen::Index r = 0; r < inputsAsMatrix.rows(); ++r)
        {
            for(Eigen::Index c = 0; c < inputsAsMatrix.cols(); ++c)
            {
                if(inputsAsMatrix.coeff(r, c) < 0)
                {
                    throw std::out_of_range("Cannot log scale if values less than 0");
                }
            }
        }
        // ADD 1 TO AVOID LOG 0 (UNDEFINED)
        inputsAsMatrix = (inputsAsMatrix.array() + 1).log();
    }
    //check no invalid errors
    if(!inputsAsMatrix.allFinite())
    {
        throw std::logic_error("INF or NaN in inputs");
    }
    // put data back in trData
    for(Eigen::Index trItemPos = 0; trItemPos < trData.size(); ++trItemPos)
    {
        trData[trItemPos].inputs = inputsAsMatrix.row(trItemPos);
    }
}

std::set<std::string> getClasses()
{
    return ClassList{CLASSES};
}

NetNumT getInputSz()
{
    return INPUT_SZ;
}

ExampleData loadTrainingDataFromFile(std::string fName)
{
    ExampleData trData;
    std::ifstream dataFile;
    dataFile.open(fName, std::ifstream::in);

    if (! dataFile.is_open())
    {
        throw std::logic_error("Could not open file");
    }

    std::string lineOfFile, delimiter = ",";
    while (std::getline(dataFile, lineOfFile))
    {
        lineOfFile += delimiter; // add delimiter to end of line so as to easily parse
        ExampleItem trItem;
        trItem.inputs.resize(1, getInputSz());


        // load expected outcome
        size_t pos = lineOfFile.find(delimiter);
        std::string targetNum = lineOfFile.substr(0, pos);
        lineOfFile.erase(0, pos + delimiter.length());

        ClassList classes = getClasses();
        std::map<ClassT, NetNumT> labels;
        for(auto it = classes.begin(); it != classes.end(); ++it)
        {
            labels[*it] = 0;
        }
        labels[targetNum] = 1;

        trItem.labels = trainingItemToVector(labels);

        // add inputs
        std::string inputAsStr;
        size_t inputCount = 0;

        while ((pos = lineOfFile.find(delimiter)) != std::string::npos)
        {
            inputAsStr = lineOfFile.substr(0, pos);
            lineOfFile.erase(0, pos + delimiter.length());
            NetNumT inputAsNum = std::stod(inputAsStr);
            trItem.inputs.row(0).col(Eigen::Index (inputCount)) << inputAsNum;
            inputCount++;
        }
        trData.push_back(trItem);
    }
    dataFile.close();
    return trData;
}

bool serialise(std::ofstream& fileOut, NNetwork& network, const ActFuncList& actFuncList)
{
    fileOut << PREFIX_ACTFUNCS;
    for (ActFunc actFunc : actFuncList)
    {
        fileOut << static_cast<std::underlying_type<ActFunc>::type>(actFunc) << ",";
    }
    fileOut << std::endl;

    fileOut << PREFIX_CLASSES;
    for(const auto& cIt : network.classes())
    {
        fileOut << cIt.first << ",";
    }
    fileOut << std::endl;

    Eigen::IOFormat WeightInit(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n", "", "", "", "");
    Eigen::IOFormat BiasInit(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "", "", "", "", "");
    fileOut << PREFIX_INPUTSZ << network.getInputs().size()  << std::endl;
    fileOut << PREFIX_BIASES << std::endl;
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        fileOut << std::fixed << network.layer(layerPos).getBiases().format(BiasInit) << std::endl;
    }
    fileOut << PREFIX_WEIGHTS << std::endl;
    for(size_t layerPos = 0; layerPos < network.numLayers(); ++layerPos)
    {
        const auto weights = network.layer(layerPos).getWeights();
        fileOut << weights.rows() << "," << weights.cols() << std::endl;
        fileOut << std::fixed << weights.format(WeightInit) << std::endl;
    }
    return true;
}

Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic> generateVectorRow(const std::string& str)
{
    size_t vecSize = std::count(str.begin(), str.end(), DELIMITER) + 1;
    Eigen::Matrix<NetNumT, 1, Eigen::Dynamic> vecToReturn;
    vecToReturn.resize(vecSize);

    std::stringstream sstream(str);
    std::string buf;
    size_t i =0;
    while (std::getline(sstream, buf, DELIMITER))
    {
        NetNumT coeff = std::stod(buf);
        vecToReturn(0, i) = coeff;
        i++;
    }
    return vecToReturn;
}

Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic> generateMatrix(std::ifstream& fileIn, size_t weightsR, size_t weightsC)
{
    std::string weightAsStr, buf;
    size_t line = 0;
    while( line < weightsR)
    {
        std::getline(fileIn, buf);
        weightAsStr += buf;
        weightAsStr += DELIMITER;
        line++;
    }
    Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic> weightsToReturn;
    weightsToReturn.resize(weightsR, weightsC);

    std::stringstream sstream(weightAsStr);
    size_t pos = 0;
    while(std::getline(sstream, buf, DELIMITER))
    {
        NetNumT coeff = std::stod(buf);

        weightsToReturn(pos/weightsC, pos % weightsC) = coeff;
        ++pos;
    }
    return weightsToReturn;

}

NNetwork deserialise(std::ifstream& fileIn, ActFuncList& actFuncList)
{
    ClassList cList;
    size_t inputSz;
    //
    std::string buf;
    size_t pos;
    std::stringstream sstream;

    //get actfuncs
    std::getline(fileIn, buf);
    pos = buf.find(PREFIX_ACTFUNCS) + std::string(PREFIX_ACTFUNCS).size();
    if(pos == std::string::npos)
    {
        throw std::logic_error("No actfuncs specified");
    }
    buf = buf.substr(pos, std::string::npos);
    sstream << buf;
    std::string actFuncStrBuf;
    while(std::getline(sstream, actFuncStrBuf, DELIMITER))
    {
        int actFuncAsNum = std::stoi(actFuncStrBuf);
        actFuncList.push_back(ActFunc(actFuncAsNum));
    }

    // classes
    std::getline(fileIn, buf);
    pos = buf.find(PREFIX_CLASSES) + std::string(PREFIX_CLASSES).size();
    if(pos == std::string::npos)
    {
        throw std::logic_error("No prefix class");
    }
    buf = buf.substr(pos, std::string::npos);
    sstream.clear();
    sstream << buf;
    std::string classesAsStr;
    while(std::getline(sstream, classesAsStr, DELIMITER))
    {
        cList.insert(classesAsStr);
    }


    //get input sz
    std::getline(fileIn, buf);
    pos = buf.find(PREFIX_INPUTSZ) + std::string(PREFIX_INPUTSZ).size();
    if(pos == std::string::npos)
    {
        throw std::logic_error("No input sz prefix");
    }
    inputSz = std::stoi (buf.substr(pos, std::string::npos));

    // construct Network
    NNetwork networkToReturn(inputSz, cList);

    //get biases and add layers
    std::getline(fileIn, buf);
    if(buf.find(PREFIX_BIASES) == std::string::npos)
    {
        throw std::logic_error("No bias prefix");
    }
    sstream.clear();
    for(size_t biasLayer = 0; biasLayer < actFuncList.size(); ++biasLayer)
    {
        std::getline(fileIn, buf);
        auto biasForLayer  = generateVectorRow(buf);
        if(biasLayer < actFuncList.size() - 1)
        {
            networkToReturn.addLayer(biasForLayer.size(), biasLayer);
        }
        networkToReturn.layer(biasLayer).setBiases(biasForLayer);
    }

    // add weights
    std::getline(fileIn, buf);
    if(buf.find(PREFIX_WEIGHTS) == std::string::npos)
    {
        throw std::logic_error("No weight prefix");
    }
    size_t weightLayerPos = 0;
    while(std::getline(fileIn, buf))
    {
        sstream.clear();
        sstream << buf;
        size_t weightRows, weightCols;

        std::getline(sstream, buf, ',');
        weightRows = stoi(buf);
        std::getline(sstream, buf);
        weightCols = stoi(buf);

        Eigen::Matrix<NetNumT, Eigen::Dynamic, Eigen::Dynamic> weightMatrix = generateMatrix(fileIn, weightRows, weightCols);
        networkToReturn.layer(weightLayerPos).setWeights(weightMatrix);
        weightLayerPos++;
    }
    return networkToReturn;
}