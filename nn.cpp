#include "vector"
#include <cmath>
#include <string>

#include "Matrix.h"
#include "NN.h"

double leakyRelu(double a) {
    return std::max(0.01 * a, a);
}

double softmax(double z, double sum) {
    return std::exp(z) / (sum + 0.000001);
};

void NeuralNetwork::addLayer(int numNodes, Activation activation) {
        int layer = W.size();
        W.push_back(Matrix(W[layer - 1].getCols(), numNodes));
        b.push_back(Matrix(numNodes, 1)); 
        g.push_back(activation);
        a.push_back(Matrix::ones(numNodes, 1));
}

void NeuralNetwork::addLayer(int numNodes, Activation activation, int input) {
    W.push_back(Matrix(input, numNodes));
    b.push_back(Matrix(numNodes, 1)); 
    g.push_back(activation);
    a.push_back(Matrix::ones(numNodes, 1));
}


void NeuralNetwork::updateBiasLayer(int layer, double val) {
        b[layer].setColToVal(0, val);
}

void NeuralNetwork::updateWeightsLayer(int layer, Matrix vals) {
        W[layer].setCol(0, vals);
    }

void NeuralNetwork::updateActivationLayer(int layer, Matrix vals) {
        a[layer].setCol(0, vals);
        // a_update.setCol(0, vals);
    }

void NeuralNetwork::printNN() {
        for (int i = 0; i < W.size(); i++) {
            std::string ac = "";
            if (g[i] == Activation::LeakyRelu) {
                ac = "LeakyRelu";
            } else if (g[i] == Activation::Softmax) {
                ac = "Softmax";
            }
            printf("Layer %i: %i nodes, %s activation\n", i+1, W[i].getRows(), ac.c_str());
        }
    }

Matrix NeuralNetwork::forwardPass(Matrix X) {

    for (int i = 0; i < W.size(); i++) {
        printf("Forward pass layer %i\n", i+1);

        Matrix a_temp = Matrix::zeros(1,1);
        if (i == 0) {
            a_temp = X;
        } else {
            a_temp = a[i-1];
        }

        a_temp.printSize();
        W[i].printSize();

        Matrix z_temp = (W[i].transpose())*a_temp + b[i];

        // z_temp.printSize();

        if (g[i] == 1) {
            // Leaky Relu
            z_temp.applyFunction(leakyRelu);
            updateActivationLayer(i, z_temp);
            z_temp.printSize();

        } else if (g[i] == 2) {
            // Softmax
            double z_sum = z_temp.expSumVec();
            z_temp.applyFunction(softmax, z_sum);
            z_temp.printSize();

            updateActivationLayer(i, z_temp);
        }
    }

    printf("Forward pass complete!\n");

    return a[a.size() - 1];
}


NeuralNetwork::NeuralNetwork(/* args */)
{
    W = std::vector<Matrix>();
    b = std::vector<Matrix>();
    g = std::vector<Activation>();
    a = std::vector<Matrix>();

}

NeuralNetwork::~NeuralNetwork()
{
}
