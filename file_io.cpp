#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
// #include <math.h>

#include "Matrix.h"


double leakyRelu(double a) {
    return std::max(0.01 * a, a);
}

double leakyReluDeriv(double x) {
    if (x >= 0) return 1;
    return 0.01;
}

double scalarMultiply(double a, double b) {
    return a * b;
}

double softmax(double z, double sum) {
    // printf("%f\n", std::exp(z));
    // printf("%f\n", std::exp(sum));
    return std::exp(z) / (sum + 0.000001);
};

Matrix oneHot(char a, std::map<char, int> alphabet) {
    Matrix vector = Matrix::zeros(alphabet.size(), 1);
    vector.set(alphabet[a],0,1.0);
    return vector;
}

double log2(double a) {
    return std::log(a);
}


double cost(Matrix &h, Matrix &y) {
    // Compute log ofo predictions
    Matrix h_log = Matrix(h);
    h_log.applyFunction(log2);

    Matrix mulRes = Matrix::elemMult(y, h_log);

    return mulRes.sumVec() * -1.0;
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
    a3.applyFunction(leakyReluDeriv);
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
            alphabet[*(charList + i)] = alphabet.size();
        } 
    }

    int alphabetSize = (int)alphabet.size();

    // ALPHABET SIZE 69 characters
    // NN needs input of 69 nodes
    // Output 69 Nodes


    // Test Forward Pass
    Matrix X = oneHot(charList[0], alphabet);
    Matrix inputLayer = Matrix(X);

    int h1Nodes = 1024;
    int h2Nodes = 4096;
    int h3Nodes = 360;


    // FORWARD PASS

    // TODO: Cache each layer's output
    Matrix theta1 = Matrix::randN(h1Nodes, alphabetSize + 1);
    inputLayer.prependVec(1.0); // Add bias
    // input.applyFunction(leakyRelu);
    // input.printMatrix();
    Matrix z1 = theta1*inputLayer;
    z1.applyFunction(leakyRelu);
    z1.printSize();
    // z1.printMatrix();

    Matrix theta2 = Matrix::randN(h2Nodes, h1Nodes);
    // z1.prependVec(1.0);
    Matrix z2 =  theta2 * z1;
    z2.applyFunction(leakyRelu);

    // z2.printMatrix();

    Matrix theta3 = Matrix::randN(h3Nodes, h2Nodes);
    // z2.prependVec(1.0);
    Matrix z3 = theta3 * z2;
    z3.applyFunction(leakyRelu);

    // z3.printMatrix();

    Matrix theta4 = Matrix::randN(alphabetSize, h3Nodes);
    // z3.prependVec(1.0);
    Matrix z4 = theta4 * z3;

    double outputSum = z4.expSumVec();

    // printf("SUM: %f\n", outputSum);
    // z4.printMatrix(); // Output
    z4.applyFunction(softmax, outputSum);
    // z4.printMatrix(); // Output

    Matrix expected = oneHot(charList[1], alphabet);

    double result = cost(z4, expected);

    printf("Cost: %f\n", result);

    std::vector<Matrix> cache = {X, z1, z2, z3, z4};
    std::vector<Matrix> W = {theta1, theta2, theta3, theta4};

    backprop(z4, expected, cache, W, 0.01);

    // Matrix abc = Matrix::ones(12, 1);

    // z4.printSize();
    // expected.printSize();



    // double result = res.sumVec();

    // printf("SUM: %f\n", result);

   

    // Cleanup
    delete[] charList;

    return 0;

}
