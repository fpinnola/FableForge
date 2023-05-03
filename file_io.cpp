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

Matrix oneHot(char a, std::map<char, int> alphabet); 
// {
//     Matrix vector = Matrix::zeros(alphabet.size(), 1);
//     vector.set(alphabet[a],0,1.0);
//     return vector;
// }

char oneHotToChar(Matrix a, std::map<char, int> alphabet) {
    int* t = new int[3];
    int highestVal = 0;
    Matrix a_cpy = Matrix(a);
    float* vals = a_cpy.getVals();
    std::sort(vals, vals + a.getRows(), std::greater<float>());

    // printf("vals[0]: %f\n", vals[0]);
    // printf("vals[1]: %f\n", vals[1]);
    // printf("vals[2]: %f\n", vals[2]);
    float thresh = 0.2;
    int ind = 0;
    for (int i = 0; i < a.getRows(); i++) {
        if(a.get(i, 0) == vals[0] || a.get(i, 0) == vals[1] || a.get(i, 0) == vals[2]) {
            t[ind] = i;
            ind+= 1;
        }
    }
    // printf("t[0]: %i\n", t[0]);
    // printf("t[1]: %i\n", t[1]);
    // printf("t[2]: %i\n", t[2]);

    // srand(time(0)); 
    int idx = (rand()%3); 
    // printf("Random generated: %i\n", idx);
    int value = t[idx];
    // printf("Looking for value: %i\n", value);
    // printf("highestVal: %i\n", highestVal);

    auto findResult = std::find_if(std::begin(alphabet), std::end(alphabet), [&](const std::pair<char, int> &pair)
    {
        // printf("%i: %c\n", pair.second, pair.first);
        return pair.second == value;
    });

    char* choices = new char[3];
    char foundKey = 'a';
    if (findResult != std::end(alphabet))
    {
        foundKey = findResult->first;
    // Now do something with the key or value!
    }

    // printf("foundKey: %c\n", foundKey);
  
    return foundKey;
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

    int epochs = 20;
    int trainingSetSize = 10000;

    printf("training set size: %i\n",trainingSetSize);
    printf("num epochs: %i\n", epochs);

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

    NeuralNetwork network2 = NeuralNetwork(alphabetSize);
    network2.addLayer(256, Activation::LeakyRelu);
    network2.addLayer(512, Activation::LeakyRelu);
    // network.addLayer(1024, Activation::LeakyRelu);
    // network.addLayer(256, Activation::LeakyRelu);
    // network.addLayer(256, Activation::LeakyRelu);
    network2.addLayer(alphabetSize, Activation::Softmax);


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
    //     // if (std::isnan(cost)){
    //     //     exit(1);
    //     // }
    //     // if (cost < 0.005){
    //     //     break;
    //     // }
    // }

    printf("GPU\n");
    for (int e = 0; e < epochs; e++) {
        time_t start = time(0);
        for (int i = 0; i < trainingSet.size(); i++) {
            Matrix X = oneHot(trainingSet[i][0], alphabet);
            Matrix expected = oneHot(trainingSet[i][1], alphabet);
            network2.trainingStepGPU(X, expected, 0.001);
        }   
        double seconds_since_start = difftime( time(0), start);
        float cost = network2.getAvgCost();
        printf("Epoch %i, avg cost: %f | time elapsed: %fs\n", e, cost, seconds_since_start);
        // if (std::isnan(cost)){
        //     exit(1);
        // }
        // if (cost < 0.005){
        //     break;
        // }
    }

    // Training Loop GPU
    // network.trainingLoopGPU(trainingSet, epochs, alphabet);

    int numOutputTokens = 80;
    char inputToken = 'a';
    char* output = (char*)malloc((numOutputTokens+1) * sizeof(char));
    output[0] = inputToken;
    printf("%c", inputToken);
    for (int i = 1; i <= numOutputTokens; i++) {
        Matrix next = network2.predict(oneHot(output[i-1], alphabet));
        char nextC = oneHotToChar(next, alphabet);
        output[i] = nextC;
        printf("%c", nextC);
    }
    printf("\n");
   
    // Cleanup
    delete[] charList;

    return 0;

}
