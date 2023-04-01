#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
// #include <vector>
#include <map>
#include <cmath>
// #include <math.h>

#include "Matrix.h"





double leakyRelu(double a) {
    return std::max(0.01 * a, a);
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

double cost2(Matrix &h, Matrix &y) {
    double out = 0.0;

    h.printDataOne();
    y.printDataOne();



    Matrix h_log = Matrix(h);
    h_log.printDataOne();
    // Matrix h_log = Matrix::copy(h);
    // h_log.printMatrix();
    h_log.applyFunction(log2);
    Matrix h_log2 = Matrix(h);
    h.printDataOne();
    h_log2.printDataOne();
    // printf("h_log2, %p\n", (void*)&h_log2);

    Matrix h_log3 = Matrix::ones(h.getRows(), h.getCols()) - h;
    h_log3.printDataOne();
    
    // h_log3.applyFunction(log2);
    // Matrix q1 = Matrix::elemMult(y, h_log);
    // q1.printSize();
    // Matrix q2 = (Matrix::ones(h.getRows(), h.getCols()) - y).elemMult(h_log2);
    
    // i1.printSize();

    // Matrix q2 = (i1 - y);
    // Matrix q3 = q2.elemMult(h_log2);

    // Matrix res = q1 + q3;

    double output = 2.2;
    printf("output: %f\n", output);

    return output;
}

double cost(Matrix &h, Matrix &y) {

    double out = 0.0;
    Matrix h_log = Matrix(h);
    // h_log.printMatrix();
    h_log.applyFunction(log2);


    // (1−𝑦_𝑡𝑟𝑎𝑖𝑛)⋅log(1−𝑦_𝑜𝑢𝑡𝑝𝑢𝑡))
    Matrix h_log2 = Matrix(h);
    Matrix h_log2_ones = Matrix::ones(h.getRows(), h.getCols());
    Matrix h_log3 = h_log2_ones - h;
    h_log3.applyFunction(log2);
    Matrix q1 = Matrix::elemMult(y, h_log);
    Matrix i1 = Matrix::ones(y.getRows(), y.getCols());
    Matrix q2 = (i1 - y);
    Matrix q3 = Matrix::elemMult(q2, h_log2);

    Matrix res = q1 + q3;
    // res.printMatrix();

    // double output = 2.2;
    double output = res.sumVec();
    printf("output: %f\n", output);

    return output;
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
    // std::set<int> alphabet;
    std::map<char, int> alphabet;



    // Add characters to set
    for (int i = 0; i < charCount; ++i) {
        if (alphabet.find(*(charList + i)) == alphabet.end()) {
            alphabet[*(charList + i)] = alphabet.size();
        } 
    }

    int alphabetSize = (int)alphabet.size();

    // printf("Alphabet size: %i\n",alphabetSize);

    // ALPHABET SIZE 69 characters
    // NN needs input of 69 nodes
    // Output 69 Nodes


    // Test Forward Pass
    Matrix inputLayer = oneHot(charList[0], alphabet);
    // printf("First char: %c index %i\n ", charList[0], alphabet[charList[0]]);
    // inputLayer.printMatrix();


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
    // z1.printMatrix();

    Matrix theta2 = Matrix::randN(h2Nodes, h1Nodes+1);
    z1.prependVec(1.0);
    Matrix z2 =  theta2 * z1;
    z2.applyFunction(leakyRelu);

    // z2.printMatrix();

    Matrix theta3 = Matrix::randN(h3Nodes, h2Nodes+1);
    z2.prependVec(1.0);
    Matrix z3 = theta3 * z2;
    z3.applyFunction(leakyRelu);

    // z3.printMatrix();

    Matrix theta4 = Matrix::randN(alphabetSize, h3Nodes+1);
    z3.prependVec(1.0);
    Matrix z4 = theta4 * z3;

    double outputSum = z4.expSumVec();

    // printf("SUM: %f\n", outputSum);
    // z4.printMatrix(); // Output
    z4.applyFunction(softmax, outputSum);
    // z4.printMatrix(); // Output

    Matrix expected = oneHot(charList[1], alphabet);
    // expected.printMatrix();
    // Matrix loss = expected - z4;
    // z4.printMatrix(); // Output

    // double result = cost(z4, expected);
    double result = cost(z4, expected);

    // printf("Cost: %f", forwardCost);

    // Matrix abc = Matrix::ones(12, 1);

    // z4.printSize();
    // expected.printSize();



    // double result = res.sumVec();

    // printf("SUM: %f\n", result);

   

    // Cleanup
    delete[] charList;

    return 0;

}
