#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
// #include <math.h>

#include "NN.h"
#include "Matrix.h"


double scalarMultiply(double a, double b) {
    return a * b;
}

Matrix oneHot(char a, std::map<char, int> alphabet) {
    Matrix vector = Matrix::zeros(alphabet.size(), 1);
    vector.set(alphabet[a],0,1.0);
    return vector;
}

double log2(double a) {
    if (a == 0.0) {
        a = 0.000001;
    }
    return std::log(a);
}

void backprop(Matrix &h, Matrix &y, std::vector<Matrix> cache, std::vector<Matrix> W, double alpha) {

    Matrix dL_da4 = h - y;

    Matrix a4 = Matrix(cache[4]);
    Matrix a3 = Matrix(cache[3]);
    Matrix a2 = Matrix(cache[2]);
    Matrix a1 = Matrix(cache[1]);

    Matrix W4 = Matrix(W[3]);
    Matrix W3 = Matrix(W[2]);
    Matrix W2 = Matrix(W[1]);
    Matrix W1 = Matrix(W[0]);

    Matrix dZ4 = a4 - y;
    Matrix dW4 = Matrix::elemMult(dZ4, a3.transpose());
    // a3.applyFunction(leakyReluDeriv);
    Matrix dZ3 = Matrix::elemMult(Matrix::elemMult(W4.transpose(), dZ4), a3);



    // Matrix dL_da3 = W[3].transpose() * dL_dz4;
    // Matrix z3 = Matrix(cache[3]);
    // z3.applyFunction(leakyReluDeriv);
    // Matrix dL_dz3 = Matrix::elemMult(dL_da3, z3);

    // Matrix dL_da2 = W[2].transpose() * dL_dz3;
    // Matrix z2 = Matrix(cache[2]);
    // z2.applyFunction(leakyReluDeriv);
    // Matrix dL_dz2 = Matrix::elemMult(dL_da2, z2);

    // Matrix dL_da1 = W[1].transpose() * dL_dz2;
    // Matrix z1 = Matrix(cache[1]);
    // z1.applyFunction(leakyReluDeriv);
    // Matrix dL_dz1 = Matrix::elemMult(dL_da1, z1);


    // dL_da4.printSize();
    // z4.printSize();


    // Matrix dL_dW4 = z4.transpose() * dL_da4;
    
    // dL_dW4.applyFunction(scalarMultiply, alpha);

    // Matrix dL_dW3 = z3.transpose() * dL_da3;
    // dL_dW3.applyFunction(scalarMultiply, alpha);

    // Matrix dL_dW2 = z2.transpose() * dL_da2;
    // dL_dW2.applyFunction(scalarMultiply, alpha);

    // Matrix dL_dW1 = z1.transpose() * dL_da1;
    // dL_dW1.applyFunction(scalarMultiply, alpha);

    // // W[3].printSize();
    // // dL_dW4.printSize();
    // W[4] = W[4] - dL_dW4;
    // W[3] = W[3] - dL_dW3;
    // W[2] = W[2] - dL_dW2;
    // W[1] = W[1] - dL_dW1;


    return;
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
    std::ifstream inputFile("input.txt");

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
    std::vector<std::vector<char>> trainingSet = generateTrainingSet(charList, datasetSize, 500);

    // Print Training Set
    // for (int i = 0; i < trainingSet.size(); i++) {
    //     printf("%c, %c\n", trainingSet[i][0], trainingSet[i][1]);
    // }


    // Test Forward Pass
    Matrix X = oneHot(charList[0], alphabet);
    Matrix expected = oneHot(charList[1], alphabet);

    Matrix inputLayer = Matrix(X);

    int h1Nodes = 1024;
    int h2Nodes = 4096;
    int h3Nodes = 360;


    NeuralNetwork network = NeuralNetwork();

    network.addLayer(h1Nodes, Activation::LeakyRelu, alphabetSize);
    network.addLayer(h2Nodes, Activation::LeakyRelu);
    // network.addLayer(2048*8, Activation::LeakyRelu);
    // network.addLayer(2048*8, Activation::LeakyRelu);
    // network.addLayer(2048*8, Activation::LeakyRelu);
    network.addLayer(h3Nodes, Activation::LeakyRelu);
    network.addLayer(alphabetSize, Activation::Softmax);

    // network.printNN();

    Matrix y_hat = network.forwardPass(X, expected);
    // y_hat.printMatrix();



    // FORWARD PASS


    // double result = cost(z4, expected);

    // printf("Cost: %f\n", result);

    // std::vector<Matrix> cache = {X, z1, z2, z3, z4};
    // std::vector<Matrix> W = {theta1, theta2, theta3, theta4};

    // backprop(z4, expected, cache, W, 0.01);

    // Matrix abc = Matrix::ones(12, 1);

    // z4.printSize();
    // expected.printSize();



    // double result = res.sumVec();

    // printf("SUM: %f\n", result);

   

    // Cleanup
    delete[] charList;

    return 0;

}
