#include "vector"

#include "Matrix.h"

enum Activation {
    LeakyRelu,
    Softmax
};

class NeuralNetwork
{
private:
    /* data */
    std::vector<Matrix> W;
    std::vector<Matrix> b;
    std::vector<Activation> g;

public:
    NeuralNetwork(/* args */);
    ~NeuralNetwork();

    void addLayer(int numNodes, Activation activation) {
        Matrix w = Matrix(numNodes, 1);
        W.push_back(w);
        Matrix b_new = Matrix(numNodes, 1);
        b.push_back(b_new); 
        g.push_back(activation);
    }

    void updateBiasLayer(int layer, double val) {
        Matrix b_update = b[layer];
        b_update.setColToVal(0, val);
    }

    void updateWeightsLayer(int layer, Matrix vals) {
        Matrix w_update = W[layer];
        w_update.setCol(0, vals);
    }
};

NeuralNetwork::NeuralNetwork(/* args */)
{
}

NeuralNetwork::~NeuralNetwork()
{
}
