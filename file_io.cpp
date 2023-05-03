#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <time.h>

#include "NN.h"
#include "Matrix.h"


float scalarMultiply(float a, float b) {
    return a * b;
}

float log2C(float a) noexcept {
    if (a == 0.0f) {
        a = 0.000001f;
    }
    return std::log(a);
}

std::vector<std::vector<char>> generateTrainingSet(char* dataset, int dataSize, int numExamples) {
    std::vector<std::vector<char>> res = std::vector<std::vector<char>>();
    for (int i = 0; i < numExamples; i++) {
        int start = std::rand()%(dataSize-0 + 1);
        std::vector<char> vals = std::vector<char>();
        vals.push_back(*(dataset + start));
        vals.push_back(*(dataset + start + 1));
        res.push_back(vals);
    }
    return res;
}

int main(int argc, char const *argv[])
{

    // READ INPUT FILE
    std::ifstream inputFile("processed.txt");

    if (!inputFile.is_open()) {
        std::domain_error("Failed ot open the input file");
        return 1;
    }

    const int maxFileSize = 1000000;
    char* charList = new char[maxFileSize];
    int charCount = 0;
    char c;

    while (inputFile.get(c)) {
        *(charList + charCount) = c;
        ++charCount;
        if (charCount == maxFileSize) {
            std::domain_error("Input file is too large");
            return 1;
        }
    }

    inputFile.close();

    // CREATE ALPHABET
    std::map<char, int> alphabet;

    // Add characters to set
    for (int i = 0; i < charCount; ++i) {
        if (alphabet.find(*(charList + i)) == alphabet.end()) {
            char v = *(charList + i);
            // printf("%c, %i\n", v, (int)v); // UNCOMMENT to print alphabet
            alphabet[*(charList + i)] = alphabet.size();
        }
    }

    // Number of unique tokens in dataset
    // Input/Output token will need this number of classes
    int alphabetSize = (int)alphabet.size();

    // Create Training Set
    int datasetSize = charCount;
    printf("datasetSize: %i\n", datasetSize);

    int epochs = 1;
    int trainingSetSize = 1;

    std::vector<std::vector<char>> trainingSet = generateTrainingSet(charList, datasetSize, trainingSetSize);

    // Print Training Set
    // for (int i = 0; i < trainingSet.size(); i++) {
    //     printf("%c, %c\n", trainingSet[i][0], trainingSet[i][1]);
    // }

    // Create NN
    NeuralNetwork network = NeuralNetwork(alphabetSize);
    network.addLayer(256, Activation::LeakyRelu);
    network.addLayer(512, Activation::LeakyRelu);
    // network.addLayer(1024, Activation::LeakyRelu);
    // network.addLayer(256, Activation::LeakyRelu);
    // network.addLayer(256, Activation::LeakyRelu);
    network.addLayer(alphabetSize, Activation::Softmax);

    network.printNN();


    // Training loop
    printf("CPU\n");
    // for (int e = 0; e < epochs; e++) {
    //     time_t start = time(0);
    //     for (int i = 0; i < trainingSet.size(); i++) {
    //         Matrix X = oneHot(trainingSet[i][0], alphabet);
    //         Matrix expected = oneHot(trainingSet[i][1], alphabet);
    //         network.trainingStep(X, expected, 0.001);
    //     }   
    //     double seconds_since_start = difftime( time(0), start);
    //     float cost = network.getAvgCost();
    //     printf("Epoch %i, avg cost: %f | time elapsed: %fs\n", e, cost, seconds_since_start);
    //     if (std::isnan(cost)){
    //         exit(1);
    //     }
    //     if (cost < 0.005){
    //         break;
    //     }
    // }

    printf("GPU\n");
    // Training Loop GPU
    network.trainingLoopGPU(trainingSet, epochs, alphabet);

   
    // Cleanup
    delete[] charList;

    return 0;

}
