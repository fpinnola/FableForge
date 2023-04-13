
#include "Matrix.h"

enum Activation {
    LeakyRelu = 1,
    Softmax = 2
};

class NeuralNetwork
{
private:
    /* data */
    std::vector<Matrix> W;
    std::vector<Matrix> b;
    std::vector<Activation> g;
    std::vector<Matrix> a;

public:
    NeuralNetwork(/* args */);
    ~NeuralNetwork();


    void addLayer(int numNodes, Activation activation);
    void addLayer(int numNodes, Activation activation, int input);

    void updateBiasLayer(int layer, double val);

    void updateWeightsLayer(int layer, Matrix vals);

    void updateActivationLayer(int layer, Matrix vals);

    void printNN();

    Matrix forwardPass(Matrix X);
};
