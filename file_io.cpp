#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>

#include "NN.h"
#include "Matrix.h"


float scalarMultiply(float a, float b) {
    return a * b;
}

Matrix oneHot(char a, std::map<char, int> alphabet) {
    Matrix vector = Matrix::zeros(alphabet.size(), 1);
    vector.set(alphabet[a],0,1.0);
    return vector;
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

    // std::vector<std::vector<float>> x1_val = {{1}, {0}}; // 0
    
    // std::vector<std::vector<float>> y1_val = {{0}, {1}}; // 1

    // std::vector<std::vector<float>> x2_val = {{0}, {1}}; // 1
    // std::vector<std::vector<float>> y2_val = {{1}, {0}}; // 0

    // Matrix X1 = Matrix(2,1, x1_val);
    // Matrix X2 = Matrix(2,1, x2_val);



    // Matrix Y1 = Matrix(2,1, y1_val);
    // Matrix Y2 = Matrix(2,1, y2_val);

    // NeuralNetwork network1(2);
    // network1.addLayer(2, Activation::LeakyRelu);
    // network1.addLayer(2, Activation::Softmax);

    // for (int i = 0; i < 1; i++) {
    //     network1.trainingStep(X1, X2);
    //     network1.trainingStep(X2, Y2);
    //     float c = network1.getAvgCost();
    //     printf("Cost: %f\n", c);
    //     // if (i % 2 == 0)
    //     //     network1.trainingStep(X2, Y2);
    //     // else 
    //     //     network1.trainingStep(X1, Y1);
    //     // if (i % 10 == 0) {
    //     //     float c = network1.getAvgCost();
    //     //     printf("Cost: %f\n", c);
    //     //     if (c < 0.005){
    //     //         break;
    //     //     }
    //     // }

    // }
    // Matrix Ytest = network1.predict(X2);
    // Matrix Ytest2 = network1.predict(X1);

    // Ytest.printMatrix();
    // Ytest2.printMatrix();
    // return 0;

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

    int epochs = 50;
    int trainingSetSize = 5000;

    std::vector<std::vector<char>> trainingSet = generateTrainingSet(charList, datasetSize, trainingSetSize);

    // Print Training Set
    // for (int i = 0; i < trainingSet.size(); i++) {
    //     printf("%c, %c\n", trainingSet[i][0], trainingSet[i][1]);
    // }

    // Create NN
    NeuralNetwork network = NeuralNetwork(alphabetSize);
    network.addLayer(256, Activation::LeakyRelu);
    network.addLayer(512, Activation::LeakyRelu);
    network.addLayer(1024, Activation::LeakyRelu);
    network.addLayer(256, Activation::LeakyRelu);
    // network.addLayer(256, Activation::LeakyRelu);
    network.addLayer(alphabetSize, Activation::Softmax);

    network.printNN();


    // Training loop
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < trainingSet.size(); i++) {
            Matrix X = oneHot(trainingSet[i][0], alphabet);
            Matrix expected = oneHot(trainingSet[i][1], alphabet);
            network.trainingStep(X, expected, 0.001);
        }   
        float cost = network.getAvgCost();
        printf("Epoch %i, avg cost: %f\n", e, cost);
        if (isnan(cost)){
            exit(1);
        }
        if (cost < 0.005){
            break;
        }
    }


    // Training loop GPU
    // for (int i = 0; i < 1; i++) {
    //     Matrix X = oneHot(trainingSet[i][0], alphabet);
    //     Matrix expected = oneHot(trainingSet[i][1], alphabet);
    //     printf("Run %i\n", i);
    //     Matrix y_hat = network.forwardPassGPU(X, expected);
    //     printf("\n");
    // }
    // network.printNN();
   

    // Cleanup
    delete[] charList;

    return 0;

}
