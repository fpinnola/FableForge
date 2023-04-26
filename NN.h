#include <tuple>
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
    std::vector<Matrix> z;
    int inputSize;

public:
    NeuralNetwork(int size);
    ~NeuralNetwork();


    void addLayer(int numNodes, Activation activation);
    void addLayer(int numNodes, Activation activation, int input);

    void updateBiasLayer(int layer, float val);

    void updateWeightsLayer(int layer, Matrix vals);

    void updateActivationLayer(int layer, Matrix vals);
    void updateLayerOutput(int layer, Matrix vals);

    void printNN();

    Matrix forwardPass(Matrix X, Matrix Y);
    Matrix forwardPassGPU(Matrix X, Matrix Y);
    void trainingStep(Matrix X, Matrix Y);
    void trainingStepGPU(Matrix X, Matrix Y);


    std::tuple<Matrix, Matrix, Matrix> backprop(int layer, Matrix dA);
};
